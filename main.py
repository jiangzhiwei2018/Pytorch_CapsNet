from model_src import model_main
from utils import random_utils
from load_data_src import load_dataset_main

if __name__ == '__main__':
    # im, la_one_hot, la = load_dataset_main.CIFARDataset(num_classes=10, download=False, train=False)[:]
    # im2, la_one_hot2, la2 = load_dataset_main.MNISTDataset(num_classes=10, download=False, train=False)[:]
    # print(im.size())
    # print(max(la))
    # print(la_one_hot.size())
    random_utils.set_seed(2)
    # model_main.MainModule(data_dir=r"data_src/CIFAR", dataset_name="CIFAR",
    #                       num_epochs=50, save_dir="./checkpoint", download=False, num_classes=100, bs=1)
    # # ds = load_dataset_main.ImageNetDataset(train=True)
    # # im, la_one_hot, la = ds[:32]
    # # print(im.size())
    # # print(la_one_hot.size())
    # # print(la.size())
    # #
    model_main.MainModule(data_dir=r"./data_src", dataset_name="MNIST",
                          num_epochs=50, save_dir="./checkpoint", download=False, num_classes=10)

    # model_main.MainModule(data_dir=r'G:/LargeDataset/ImageNet/2012', dataset_name="ImageNet",
    #                       num_epochs=50, save_dir="./checkpoint", download=False, num_classes=1000, bs=1)

