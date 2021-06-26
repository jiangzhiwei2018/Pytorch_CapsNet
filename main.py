from model_src import model_main
from utils import random_utils

if __name__ == '__main__':
    random_utils.set_seed(2)
    model_main.MainModule(num_epochs=50, save_dir="./checkpoint", download=False)

