from load_data_src import load_dataset_main
from model_src import model_main
from utils import random_utils

if __name__ == '__main__':
    random_utils.set_seed(2)
    # load_dataset_main.load_mnist_dataset()
    # load_dataset_main.MyMnistDataset()
    model_main.MainModule()

