"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.hl1 = nn.Linear(2, hid)
        self.hl2 = nn.Linear(hid, hid)
        self.out_layer = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.hl1(input))
        self.hid2 = torch.tanh(self.hl2(self.hid1))
        output = torch.sigmoid(self.out_layer(self.hid2))
        return output

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.hl1 = nn.Linear(2, hid)
        self.hl2 = nn.Linear(hid, hid)
        self.hl3 = nn.Linear(hid, hid)
        self.out_layer = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.hl1(input))
        self.hid2 = torch.tanh(self.hl2(self.hid1))
        self.hid3 = torch.tanh(self.hl3(self.hid2))
        output = torch.sigmoid(self.out_layer(self.hid3))
        return output

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.hl1 = nn.Linear(2, num_hid)
        self.hl2 = nn.Linear(num_hid + 2 , num_hid)
        self.out_layer = nn.Linear(num_hid * 2 + 2, 1)

    def forward(self, input):
        hl1 = self.hl1(input)
        self.hid1 = torch.tanh(hl1)

        hl2 = self.hl2(torch.cat([input, self.hid1], 1))
        self.hid2 = torch.tanh(hl2)
        
        ol = self.out_layer(torch.cat([input, self.hid1, self.hid2], 1))
        output = torch.sigmoid(ol)
        return output
