"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.l1 = nn.Linear(28*28, 10)

    def forward(self, x):
        print(x.shape)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = F.log_softmax(self.l1(x), dim=1)
        return x 

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        hiden_nodes = 150
        self.ll1 = nn.Linear(28*28, hiden_nodes)
        self.ll2 = nn.Linear(hiden_nodes, 10)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.ll1(x)
        # print(x.shape)
        x = torch.tanh(x)
        x = self.ll2(x)
        # print(x.shape)
        x = F.log_softmax(x)
        return x

class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.cl1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, padding=2)
        self.cl2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=4, padding=2)
        self.ll1 = nn.Linear(in_features=32 * 8 * 8, out_features=600, bias=True)
        self.ll2 = nn.Linear(in_features=600, out_features=10, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=4, padding=2)

    def forward(self, x):
        x = self.cl1(x)
        x = F.relu(x)
        x = self.cl2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.ll1(x)
        x = F.relu(x)
        x = self.ll2(x)
        x = F.log_softmax(x, dim=1)

        return x
