import torch
from torchvision import transforms, datasets
from data_aug.gaussian_blur import GaussianBlur
from tqdm import tqdm


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor()])
    return data_transforms


def data_test():
    folder_path = '/Users/zhangce/Desktop/handout/SimCLR/datasets'
    batch_size = 256
    n_views = 2
    # train_dataset = datasets.CIFAR10(folder_path, train=True,
    #                                  transform=ContrastiveLearningViewGenerator(
    #                                      get_simclr_pipeline_transform(32),
    #                                      n_views),
    #                                  download=True)
    '''
    train_dataset[i]: tuple (list(tensor...), int)
    '''

    train_dataset = datasets.CIFAR10(folder_path, train=True, download=True)
    '''
    train_dataset[i]: tuple (PIL.image, int)
    '''
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True,
    #     pin_memory=True, drop_last=True)
    # for data, label in train_loader:
    #     # data: list(2)
    #     print(data[0].size(), label.size())
    #     break


def tqdm_test():
    ls = [0 for _ in range(100)]
    for l, i in enumerate(tqdm(ls)):
        print(l, i)

if __name__ == '__main__':
    data_test()
    # tqdm_test()