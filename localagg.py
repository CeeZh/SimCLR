import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import faiss

from utils import save_config_file, save_checkpoint, AverageMeter

torch.manual_seed(0)

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


DEFAULT_KMEANS_SEED = 1234


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


class MemoryBank(object):
    """For efficiently computing the background vectors."""

    def __init__(self, size, dim, value=None):
        self.size = size
        self.dim = dim
        self.bank = self._create() if value is None else value
        # print(colored('Warning: using in-place scatter in memory bank update function', 'red'))

    def _create(self):
        # initialize random weights
        mb_init = torch.rand(self.size, self.dim)
        std_dev = 1. / np.sqrt(self.dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        # L2 normalise so that the norm is 1
        mb_init = l2_normalize(mb_init, dim=1)
        return mb_init.detach()  # detach so its not trainable

    def get_all_dot_products(self, vec):
        # [bs, dim]
        assert len(vec.size()) == 2
        return torch.matmul(vec, torch.transpose(self.bank, 1, 0))

    def get_dot_products(self, vec, idxs):
        vec_shape = list(vec.size())  # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self.bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self.bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self.bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)

    def update(self, indices, data_memory):
        # in lieu of scatter-update operation
        data_dim = data_memory.size(1)
        data_memory = data_memory.detach()
        indices = indices.unsqueeze(1).repeat(1, data_dim)
        self.bank = self.bank.scatter_(0, indices, data_memory)


def run_kmeans(x, nmb_clusters, verbose=False,
               seed=DEFAULT_KMEANS_SEED, gpu_device=0):
    """
    Runs kmeans on 1 GPU.

    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters

    Returns:
    --------
    list: ids of data in each cluster
    """
    n_data, d = x.shape

    # niter = 20
    # kmeans = faiss.Kmeans(d, nmb_clusters, niter=niter, verbose=verbose, gpu=True)
    # kmeans.train(x)
    # _, I = kmeans.index.search(x, 1)
    # return [int(n[0]) for n in I], 0

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = gpu_device

    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


def compute_clusters(k, memory_bank, gpu_device=0):
    """
    Performs many k-means clustering.

    Args:
        x_data (np.array N * dim): data to cluster
    """
    data = memory_bank.cpu().detach().numpy()
    pred_labels = []
    for k_idx, each_k in enumerate(k):
        # cluster the data
        I, _ = run_kmeans(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                          gpu_device=gpu_device)
        clust_labels = np.asarray(I)
        pred_labels.append(clust_labels)
    pred_labels = np.stack(pred_labels, axis=0)
    pred_labels = torch.from_numpy(pred_labels).long()
    return pred_labels


class LocalAggregationLossModule(torch.nn.Module):

    def __init__(self, memory_bank, cluster_label, k=4096, t=0.07, m=0.5):
        super(LocalAggregationLossModule, self).__init__()
        self.k, self.t, self.m = k, t, m
        self.indices = None
        self.outputs = None
        self.memory_bank = memory_bank
        self.cluster_label = cluster_label
        self.data_len = memory_bank.size(0)

    def _softmax(self, dot_prods):
        Z = 2876934.2 / 1281167 * self.data_len
        return torch.exp(dot_prods / self.t) / Z

    def updated_new_data_memory(self, indices, outputs):
        outputs = l2_normalize(outputs)
        data_memory = torch.index_select(self.memory_bank, 0, indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * outputs
        return l2_normalize(new_data_memory, dim=1)

    def _get_all_dot_products(self, vec):
        assert len(vec.size()) == 2
        return torch.matmul(vec, torch.transpose(self.memory_bank, 1, 0))

    def __get_close_nei_in_back(self, each_k_idx, cluster_labels,
                                back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][self.indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = batch_labels.unsqueeze(1).expand(-1, k)
        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei.byte()

    def __get_relative_prob(self, all_close_nei, back_nei_probs):
        relative_probs = torch.sum(
            torch.where(
                all_close_nei,
                back_nei_probs,
                torch.zeros_like(back_nei_probs),
            ), dim=1)
        # normalize probs
        relative_probs = relative_probs / torch.sum(back_nei_probs, dim=1, keepdim=True)
        return relative_probs

    def forward(self, indices, outputs):
        """
        :param back_nei_idxs: shape (batch_size, 4096)
        :param all_close_nei: shape (batch_size, _size_of_dataset) in byte
        """
        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)

        k = self.k

        all_dps = self._get_all_dot_products(self.outputs)
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=k, sorted=False, dim=1)
        back_nei_probs = self._softmax(back_nei_dps)

        all_close_nei_in_back = None
        no_kmeans = self.cluster_label.size(0)
        with torch.no_grad():
            for each_k_idx in range(no_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(
                    each_k_idx, self.cluster_label, back_nei_idxs, k)

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    # assuming all_close_nei and curr_close_nei are byte tensors
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        relative_probs = self.__get_relative_prob(all_close_nei_in_back, back_nei_probs)
        loss = -torch.mean(torch.log(relative_probs + 1e-7))

        # compute new data memory
        new_data_memory = self.updated_new_data_memory(self.indices, self.outputs)

        return loss, new_data_memory


class LocalAggregation(object):

    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.dataset = kwargs['dataset']
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.learning_rate,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True, drop_last=True)
        self.first_iteration_kmeans = True
        self.init_memory_bank()
        self.init_cluster_label()
        self.loss_fn = LocalAggregationLossModule(self.memory_bank.bank, self.cluster_label,
                                                  self.args.k, self.args.t, self.args.m)
        self.writer = SummaryWriter(log_dir=kwargs['log_dir'])
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def init_memory_bank(self):
        all_data = []
        for _, image, _ in self.dataset:
            all_data.append(np.array(image))
        self.model.eval()
        all_data = torch.from_numpy(np.array(all_data))
        all_vec = self.model(all_data)
        self.model.train()
        self.memory_bank = MemoryBank(len(self.dataset), self.args.out_dim, all_vec)

    def init_cluster_label(self):
        # initialize cluster labels
        k = [self.args.kmeans_k for _ in range(self.args.n_kmeans)]
        cluster_label = compute_clusters(k, self.memory_bank.bank, self.args.device)
        self.cluster_label = cluster_label

    def train(self, train_loader):
        if apex_support and self.args.fp16_precision:
            logging.debug("Using apex for fp16 precision training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2',
                                                        keep_batchnorm_fp32=True)
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start LocalAggregation training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        min_loss = float('inf')
        for epoch_counter in range(self.args.epochs):
            epoch_loss = AverageMeter()
            for batch_i, (indices, images, _) in enumerate(tqdm(train_loader)):
                images = images.to(self.args.device)  # torch.Size([batch_size, 3, 32, 32])

                features = self.model(images)  # torch.Size([batch_size, out_dim])

                loss, new_data_memory = self.loss_fn(indices, features)
                epoch_loss.update(loss.data)

                self.optimizer.zero_grad()
                if apex_support and self.args.fp16_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)

                n_iter += 1

                with torch.no_grad():
                    self.memory_bank.update(indices, new_data_memory)

                    if self.first_iteration_kmeans or batch_i % self.args.kmeans_freq == 0:

                        if self.first_iteration_kmeans:
                            self.first_iteration_kmeans = False

                        # get kmeans clustering (update our saved clustering)
                        k = [self.args.kmeans_k for _ in range(self.args.n_kmeans)]
                        self.cluster_label = compute_clusters(k, self.memory_bank.bank, self.args.gpu_index)

            if epoch_counter > 10 and min_loss > epoch_loss.avg:
                min_loss = epoch_loss.avg
                checkpoint_name = 'best_model.pth.tar'
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {epoch_loss.avg}")
        logging.info("LocalAggregation Training has finished.")
