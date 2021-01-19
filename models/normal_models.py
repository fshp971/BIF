import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from . import bayes_nn
from .bayes_nn import normalLinear as Linear
from .bayes_nn import normalConv2d as Conv2d
# from .bayes_nn import normalBatchNorm2d as BatchNorm2d


class normalArch(nn.Module):
    def __init__(self):
        super(normalArch, self).__init__()
        self.__dict__['_normal_modules'] = dict()

    def __setattr__(self, name, value):
        if isinstance(value, bayes_nn.normalModule):
            self.__dict__['_normal_modules'][name] = value
        super(normalArch, self).__setattr__(name, value)

    def kl_loss(self):
        # return 0
        res = 0
        for name, mod in self.__dict__['_normal_modules'].items():
            res += mod.kl_loss()
        return res


class normalMLP(normalArch):
    def __init__(self, prior_sig=None, in_dims=1):
        super(normalMLP, self).__init__()

        self.linear1 = Linear(prior_sig, 32*32*in_dims, 512)
        self.linear2 = Linear(prior_sig, 512, 512)
        self.linear3 = Linear(prior_sig, 512, 10)

    def forward(self, x):
        x = self.linear1( x.view(len(x),-1) )
        x = F.relu(x)
        x = self.linear2( x.view(len(x),-1) )
        x = F.relu(x)
        x = self.linear3( x.view(len(x),-1) )
        return x


''' standard LeNet-5 (for 32*32 input) '''
class normalLeNet(normalArch):
    def __init__(self, prior_sig=None, in_dims=1):
        super(normalLeNet, self).__init__()

        self.conv1 = Conv2d(prior_sig, in_dims, 6, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(prior_sig, 6, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = Linear(prior_sig, 5*5*16, 120)
        self.fc2 = Linear(prior_sig, 120, 84)
        self.fc3 = Linear(prior_sig, 84, 10)

    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ) )
        x = self.pool2( F.relu( self.conv2(x) ) )
        x = F.relu( self.fc1(x.view(len(x), -1)) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)
        return x
