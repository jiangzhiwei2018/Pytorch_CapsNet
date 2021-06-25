import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import math


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def cal_normal(v, dim=-1, keepdim=False):
    """

    :return:
    """
    normal = (torch.sum(v ** 2, dim=dim, keepdim=keepdim)) ** 0.5
    return normal


def squash(sr, dim=1):
    """

    :param dim:
    :param sr:(bs, dim)
    :return:
    """
    sr_normal = cal_normal(sr, keepdim=True, dim=dim)
    sr_normal2 = sr_normal ** 2
    v = (sr / sr_normal) * (sr_normal2 / (1 + sr_normal2))
    return v  # (bs, nums, dim)


def dynamic_routing(u, br):
    """
    u: (b, num_size, num_classes, dim)
    br: (b, num_size, num_classes, 1)
    :return:
    """
    cr = F.softmax(br, dim=1)  # b (b, num_size, num_classes, 1)
    sr = torch.sum(cr * u, dim=1)  # cr*u(b, num_size, num_classes, dim)  sr(b, num_classes, dim)
    vr = squash(sr, dim=-1)  # (b, num_classes, dim)
    # torch.transpose(u, 1, 2)
    sm = torch.einsum('bncd,bcd->bnc', u, vr).unsqueeze(dim=3)
    # print(sm.size())
    br = br + sm
    return br, vr


class Margin_loss(object):
    def __init__(self, lam=0.5, positive_m=0.9, negative_m=0.1):
        """
        负数
        """
        self.lam = lam
        self.positive_m = positive_m
        self.negative_m = negative_m

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, pred, labels, **kwargs):
        """

        :param pred: (b, num_cls, dim)
        :param labels: (b, num_cls)
        :param kwargs:
        :return:
        """
        # print(pred.size())
        # print(labels.size())
        dist = cal_normal(pred, dim=-1)  # (b, num_cls)
        # print(dist.size())
        # labels =
        L = labels * (torch.relu(self.positive_m - dist) ** 2) + \
            self.lam * (1 - labels) * (torch.relu(dist - self.negative_m) ** 2)
        loss = torch.sum(L, dim=-1)
        return torch.mean(loss)


class BaseCapsuleLayer(nn.Module):
    def __init__(self, in_channel, out_dim, num_routing_iterations=2):
        super(BaseCapsuleLayer, self).__init__()

    def forward(self, inx):
        """

        :return:
        """


class PrimaryCapsules(nn.Module):
    def __init__(self, num_conv_units=8, **kwargs):
        super(PrimaryCapsules, self).__init__()
        self.conv_list = nn.ModuleList([nn.Conv2d(**kwargs) for i in range(num_conv_units)])

    def forward(self, inx):
        """

        :return:
        """
        all_conv_result = []
        for idx, n_conv in enumerate(self.conv_list):
            all_conv_result.append(n_conv(inx).unsqueeze(dim=0))
        out = torch.cat(all_conv_result, dim=0)  # (32, bs, 8, 6, 6)
        out = out.permute(1, 0, -2, -1, -3)
        return out


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, nums=32 * 6 * 6, num_classes=10, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(nums, num_classes, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(nums, num_classes, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x):
        """

        :param input_x: (bs, nums, in_features)
        :return:
        """
        assert input_x.size(1) == self.weight.size(0)
        # print(self.weight.size())
        z = torch.einsum('bnm,ncmo->bnco', input_x, self.weight)  # compare torch.nn.functional.bilinear
        return z + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class RoutingBase(nn.Module):
    def __init__(self, num_routing_iterations=1, **kwargs):
        super(RoutingBase, self).__init__()
        self.num_routing_iterations = num_routing_iterations
        # self.b = torch.zeros(size=(num_size, num_classes, 1))
        # self.weight = Parameter(torch.Tensor(1))

    def forward(self, inx):
        """
        inx: (b, num_size, num_classes, dim)
        :return:
        """
        # vr (b, num_classes, dim)
        v_h = []
        b_h = []
        inx_device = inx.device
        br = torch.zeros(size=(*inx.size()[:-1], 1), requires_grad=False, device=inx_device)
        for i in range(self.num_routing_iterations):
            br, vr = dynamic_routing(inx, br)
            v_h.append(vr.unsqueeze(dim=3))
            b_h.append(br)
        return torch.cat(b_h, dim=-1), torch.cat(v_h, dim=-1)


# class RoutingLayer(nn.Module):
#     def __init__(self, num_routing_iterations=2, num_size=32 * 6 * 6, in_dm=8, out_dim=16, **kwargs):
#         super(RoutingLayer, self).__init__()
#         self.mul_linear = MyLinear(in_features=in_dm, out_features=out_dim, nums=num_size)
#         self.routing_base = RoutingBase(num_routing_iterations=num_routing_iterations, num_size=num_size)
#
#     def forward(self, inx):
#         """
#
#         :return: (b, 32, 6, 6, 8)
#         """
#         x = inx.flatten(start_dim=1, end_dim=3)
#         x = self.mul_linear(x)
#         return x


class BaseCapsuleNet(nn.Module):
    def __init__(self, in_channel=1, out_dims=16, num_classes=10,
                 hidden_channels=8, num_routing_iterations=2,
                 num_primary_conv_units=32):
        super(BaseCapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=256,
                               kernel_size=(9, 9),
                               stride=(1, 1))
        self.primary_capsules = PrimaryCapsules(in_channels=256, out_channels=hidden_channels,
                                                kernel_size=9, stride=2,
                                                num_conv_units=num_primary_conv_units,
                                                groups=int(256 / 32))
        self.hidden_channels = hidden_channels
        self.out_dims = out_dims
        self.num_classes = num_classes
        self.num_size = 32 * 6 * 6
        self.my_linear = MyLinear(in_features=hidden_channels,
                                  out_features=out_dims,
                                  num_classes=num_classes,
                                  nums=self.num_size)
        self.routing_base = RoutingBase(num_routing_iterations=num_routing_iterations)

    def forward(self, inx):
        """

        :return:
        """
        out1 = torch.relu(self.conv1(inx))
        out1 = self.primary_capsules(out1)
        out1 = out1.flatten(1, 3)
        # if self.num_size != out1.size(1):
        #     self.num_size = out1.size(1)
        #     now_device = self.my_linear.weight.device
        #     self.my_linear = MyLinear(in_features=self.hidden_channels,
        #                               out_features=self.out_dims,
        #                               num_classes=self.num_classes,
        #                               nums=self.num_size).to(device=now_device)
        out1 = self.my_linear(out1)
        # print(out1.size())
        out1 = self.routing_base(out1)
        return out1


if __name__ == '__main__':
    n = 5  # 类别数
    my_linear = MyLinear(8, 16)
    # indices = torch.randn(size=(15, 256, 20, 20))  # 生成数组元素0~5的二维数组（15*15）
    # # one_hot = F.one_hot(indices, n)  # size=(15, 15, n)
    # # bsnet = BaseCapsuleNet()
    # bsnet = PrimaryCapsules(in_channels=256, out_channels=8, kernel_size=9, stride=2, num_conv_units=32,
    #                         groups=int(256 / 32))
    #
    # print(bsnet(indices).size())
    # x1 = torch.randn(size=(15, 10))
    # x1 = torch.Tensor(10, 20, 30)
    # print(x1)
    # my = MyLinear(in_features=8, out_features=16)
    x1 = torch.randn(size=(64, 1, 28, 28)).to(device=device)
    # y1 = torch.randn(size=(1152, 10, 8, 16))
    base_capsule = BaseCapsuleNet().to(device=device)
    print(base_capsule(x1)[1].size())
    # z1 = torch.randn(size=(1152, 10, 16))
    # print(torch.bmm(x1, y1).size())
    # print(torch.einsum('bnm,ncmo->bnco', x1, y1)+z1)
    # my_linear(x)
    # z1 = torch.einsum('bnm,nma->bna', x, y)  # compare torch.nn.functional.bilinear
    # l = MyLinear(in_features=8, out_features=16)
    # l1 = RoutingLayer()
    # z1 = my(x1)
    # z1 = torch.randint(low=1, high=2, size=(64, 1152, 10, 16))
    # z1 = torch.rand(size=(64, 1152, 10, 16))
    # # z2 = torch.zeros(size=(*z1.size()[:-1], 1))
    # y1 = torch.randint(low=1, high=2, size=(64, 10, 16))
    # b = torch.randint(low=1, high=2, size=(64, 1152, 10, 1))
    # z = torch.einsum('bncd,bcd->bnc', z1, y1).unsqueeze(dim=3)
    # print(z.size())
    # roouting = RoutingBase()
    # roouting(z1[0][-1])
    # print(roouting(z1)[0].size())
    # print(roouting(z1)[1].size())
    # print(roouting(z1)[0][0])
    # all_z = []
    # for i in range(z1.size(0)):
    #     """
    #     """
    #     nz = z1[i]
    #     ny = y1[i]
    #     all_nz = []
    #     for j in range(10):
    #         nz_j = nz[:, j]
    #         ny_j = ny[j]
    #         # print(nz_j.size())
    #         # print(ny_j.size())
    #         # print(torch.matmul(nz_j, ny_j).size())
    #         all_nz.append(torch.matmul(nz_j, ny_j).unsqueeze(dim=1))
    #     all_nz = torch.cat(all_nz, dim=1).unsqueeze(dim=0)
    #     all_z.append(all_nz)
    #     # break
    #     # print(torch.matmul(nz, ny).size())
    # all_z = torch.cat(all_z, dim=0)
    # print(all_z - z)
    # print(z2.size())
    # print(torch.zeros(size=(*z1.size()[:-1], 1))
    # sm = torch.einsum('bncd,bcd->b', u, vr)
    # print(z2.size())
    # b (1152, 10)
    # z = []
    # for i in range(1152):
    #     n_x = x[:, i]
    #     n_y = y[i]
    #     z.append(torch.matmul(n_x, n_y).unsqueeze(dim=1))
    #     print(z.size())
    # print(z.size())
    # z = torch.cat(z, dim=1)
    # print(z-z1)
