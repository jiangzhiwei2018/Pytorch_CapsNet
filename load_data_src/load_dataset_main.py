from typing import Union
import os
from torchvision.transforms import InterpolationMode

from load_data_src import imagenet_loaders
import torchvision
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
# import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import cv2


to_pil_image = transforms.ToPILImage()
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")


class MNISTDataset(Dataset):
    def __init__(self, data_dir=r"./data_src", train=True, download=False, num_classes=10):
        """

        :param data_dir:
        :param train:
        """
        assert num_classes == 10
        trans = transforms.Compose(
            [
                # transforms.ToTensor()
                transforms.Normalize((0.5,), (0.5,))
            ])
        dataset = torchvision.datasets.MNIST(  # torchvision可以实现数据集的训练集和测试集的下载
            root=data_dir,  # 下载数据，并且存放在data文件夹中
            train=train,  # train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True
            # ，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分。
            transform=None,  # 数据的标准化等操作都在transforms中，此处是转换
            download=download  # 瞎子啊过程中如果中断，或者下载完成之后再次运行，则会出现报错
        )
        self.data_images = trans(dataset.data.unsqueeze(dim=1).float() / 255)
        self.data_labels = dataset.targets
        self.data_labels_one_hot = F.one_hot(dataset.targets, 10)
        self.len = self.data_images.size(0)
        # print(self.data_images[0])

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        return self.data_images[index], self.data_labels_one_hot[index], self.data_labels[index]

    def __len__(self):
        return self.len


# def load_cifar100(path):
#     with open(path, 'rb') as f:
#         data = pickle.load(f, encoding='bytes')
#         labels = data[b"fine_labels"]
#         images = data[b"data"]
#         return labels, images


class CIFARDataset(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, data_dir=r"data_src/CIFAR", train=True, download=True, num_classes: Union[int] = 10):
        # super(CifarDataset, self).__init__()
        assert num_classes in [10, 100]
        if num_classes == 10:
            cifar_dataset = torchvision.datasets.CIFAR10(
                root=data_dir,
                train=train,
                download=download
            )
        else:
            cifar_dataset = torchvision.datasets.CIFAR100(
                root=data_dir,
                train=train,
                download=download
            )
        trans = transforms.Compose(
            [
                # transforms.ToTensor()
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.data_images = trans(torch.from_numpy(cifar_dataset.data).float().permute(0, 3, 1, 2) / 255)
        self.data_labels = torch.tensor(cifar_dataset.targets)
        self.data_labels_one_hot = F.one_hot(self.data_labels, num_classes)
        self.len = self.data_images.size(0)
        # print(trans(self.data_images[0]))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_images[index], self.data_labels_one_hot[index], self.data_labels[index]


class ImageNetDataset(Dataset):
    def __init__(self, data_dir=r'G:/LargeDataset/ImageNet/2012',
                 train=False, num_classes: Union[int] = 1000, **kwargs):
        """

        :param data_dir:
        :param train:
        """
        cache_dir = os.path.join(data_dir, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.save_pth = os.path.join(cache_dir, f"train={train}_ImageNet_dataset.pt")
        if os.path.exists(self.save_pth):
            print("local loading")
            self.image_dataset = torch.load(self.save_pth)
        else:
            print("new loading")
            self.image_dataset = imagenet_loaders.get_imagenet(root=data_dir, train=train,
                                                               transform=None, target_transform=None)
            torch.save(self.image_dataset, self.save_pth)
        # self.imges = np.array(self.image_dataset.imgs)
        # print(self.imges)
        self.data_labels = torch.tensor(self.image_dataset.targets)
        # self.data_labels_one_hot = F.one_hot(self.data_labels, num_classes)
        self.num_classes = num_classes
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
                # transforms.CenterCrop(224),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # to_pil_image
            ])
        self.len = len(self.data_labels)

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        if isinstance(index, int):
            pth_list = [self.image_dataset.imgs[index]]
        else:
            pth_list = self.image_dataset.imgs[index]
        imgs = self.read_batch_imgs(pth_list=pth_list)
        return imgs, F.one_hot(self.data_labels[index], self.num_classes), self.data_labels[index]

    def read_batch_imgs(self, pth_list):
        """

        :param pth_list:
        :return:
        """
        bs_imgs = []
        for n_img_inf in pth_list:
            imgs = self.trans(cv2.imread(n_img_inf[0]))[None]
            # print(imgs)
            # plt.imshow(imgs)
            # plt.show()
            # imgs2 = cv2.imread(n_img_inf[0])[:]
            # cv2.imshow("before", imgs2)
            # # imgs2 = cv2.resize(imgs2, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            # # print(imgs2.shape)
            # cv2.imshow("after", imgs.numpy())
            bs_imgs.append(imgs)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow(imgs)
            # print(imgs.shape)
        if len(bs_imgs) == 1:
            return bs_imgs[0][0]
        return torch.cat(bs_imgs, dim=0)

    def __len__(self):
        return self.len

# def load_mnist_dataset(data_dir=r"./data_src/MNIST"):
#     """
#
#     :return:
#     """
#     trainDataset = torchvision.datasets.MNIST(  # torchvision可以实现数据集的训练集和测试集的下载
#         root=data_dir,  # 下载数据，并且存放在data文件夹中
#         train=True,  # train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分。
#         transform=transforms.ToTensor(),  # 数据的标准化等操作都在transforms中，此处是转换
#         download=False  # 瞎子啊过程中如果中断，或者下载完成之后再次运行，则会出现报错
#     )
#     testDataset = torchvision.datasets.MNIST(
#         root=data_dir,
#         train=False,
#         transform=transforms.ToTensor(),
#         download=False
#     )
#     images = testDataset.data.unsqueeze(dim=3)
#     labels = testDataset.targets
#     # print(images[0].tolist())
#     print(images.shape)
#     plt.imshow(images[0], cmap='gray')
#     plt.axis('off')  # 关掉坐标轴为 off
#     plt.show()
#     # img = to_pil_image(images[0])
#     # img.show()


# if __name__ == '__main__':
# load_mnist_dataset()
# cifar100 = Cifar100(dirname, train=True)
