import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from utils import random_utils
import matplotlib.pyplot as plt
from model_src import bease_capsuleNet
from load_data_src import load_dataset_main
import copy
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class MainModule(nn.Module):

    def __init__(self, data_dir=r"./data_src", bs=32, num_epochs=10, save_dir="./checkpoint"):
        super(MainModule, self).__init__()
        self.num_epochs = num_epochs
        self.train_dl = DataLoader(load_dataset_main.MyMnistDataset(data_dir=data_dir, train=True), shuffle=True,
                                   batch_size=bs)
        self.test_dl = DataLoader(load_dataset_main.MyMnistDataset(data_dir=data_dir, train=False), shuffle=True,
                                  batch_size=bs)
        self.net = bease_capsuleNet.BaseCapsuleNet().to(device=device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.001)
        self.loss_func = bease_capsuleNet.Margin_loss()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_pth = os.path.join(save_dir, "final_model.pt")
        self.model_fit()

    def model_fit(self):
        """

        :return:
        """

        def eval_func(net, data_ld):
            all_loss = 0
            cnt = 0
            all_acc = 0
            for idx, (batch_x, batch_y_one_hot, batch_y_label) in enumerate(data_ld):
                batch_x, batch_y_one_hot, batch_y_label = \
                    batch_x.to(device), batch_y_one_hot.to(device), \
                    batch_y_label.cpu().detach().numpy()
                b_h, v_h = net(batch_x)
                v_h = v_h[..., -1]
                # reg_pre_out = torch.abs(reg_pre_out)
                # clf_loss = self.multi_label_loss(clf_pre_out, batch_y_encode)
                loss = self.loss_func(v_h, batch_y_one_hot)
                # reg_loss = self.mse_loss(reg_pre_out, batch_y)
                all_loss += float(loss.data)
                # all_reg_loss += float(reg_loss.data)
                pred_labels = self.predict(batch_x).cpu().detach().numpy()
                acc = cal_acc(pred_labels, batch_y_label)
                all_acc += acc
                cnt = idx+1
            return all_loss/cnt, all_acc/cnt

        min_loss = float("inf")
        all_cnt = 0
        for epoch in range(self.num_epochs):
            for train_idx, (train_batch_x, train_batch_y_one_hot, train_batch_y_label) in enumerate(self.train_dl):
                self.net.train()
                train_batch_x, train_batch_y_one_hot, train_batch_y_label = \
                    train_batch_x.to(device), train_batch_y_one_hot.to(device), \
                    train_batch_y_label.cpu().detach().numpy()

                train_b_h, train_v_h = self.net(train_batch_x)
                train_v_h = train_v_h[..., -1]
                train_loss = self.loss_func(train_v_h, train_batch_y_one_hot)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.net.eval()
                print(f"epoch={epoch}, train_loss={train_loss.data}")
                # all_cnt = all_cnt + 1
                # if all_cnt % 10 != 0:
                #     continue
            torch.save(self.net.to(device=torch.device("cpu")), self.save_pth)
            self.net.to(device=device)
            with torch.no_grad():
                test_loss, test_acc = eval_func(self.net, self.test_dl)
                print(f"test_loss:{test_loss}, test_acc={test_acc}")

    def predict(self, inx):
        """

        :return:
        """
        inx = inx.to(device=device)
        self.net = self.net.to(device=device)
        b_h, v_h = self.net(inx)
        v = v_h[..., -1]
        v = bease_capsuleNet.cal_normal(v, dim=-1, keepdim=False)
        soft_max = F.softmax(v, dim=-1)
        arg_max = torch.argmax(soft_max, dim=-1, keepdim=False)
        return arg_max


def cal_acc(pred, real_label):
    """

    :param pred:
    :param real_label:
    :return:
    """
    return np.equal(pred, real_label).mean()





