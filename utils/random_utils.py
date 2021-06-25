import numpy as np
import random
import torch
import os


use_cuda = torch.cuda.is_available()
# print(use_cuda)


def set_seed(seed=2):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


