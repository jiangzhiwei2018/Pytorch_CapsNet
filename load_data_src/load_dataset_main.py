import torchvision
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
# import matplotlib.pyplot as plt


to_pil_image = transforms.ToPILImage()
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")


class MyMnistDataset(Dataset):
    def __init__(self, data_dir=r"./data_src", train=True, download=False):
        """

        :param data_dir:
        :param train:
        """
        trans = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        dataset = torchvision.datasets.MNIST(  # torchvision可以实现数据集的训练集和测试集的下载
            root=data_dir,  # 下载数据，并且存放在data文件夹中
            train=train,  # train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True
            # ，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分。
            transform=trans,  # 数据的标准化等操作都在transforms中，此处是转换
            download=download  # 瞎子啊过程中如果中断，或者下载完成之后再次运行，则会出现报错
        )
        self.data_images = dataset.data.unsqueeze(dim=1).float()
        self.data_labels = dataset.targets
        self.data_labels_one_hot = F.one_hot(dataset.targets, 10)
        self.len = self.data_images.size(0)

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        return self.data_images[index], self.data_labels_one_hot[index], self.data_labels[index]

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
#     load_mnist_dataset()


