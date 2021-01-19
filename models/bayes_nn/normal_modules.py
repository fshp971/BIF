import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch._six import container_abcs
from itertools import repeat


class normalModule(nn.Module):
    ''' This is the base module to support Bayesian neural network (BNN)
    with mean-field Gaussian variational family.

    Methods:
        kl_loss: Calculate the KL divergence between prior `p` and
                 variational distribution `q`. It will only sample
                 for once every time it is called, so one have to
                 repeatly calling it to achieve sufficient accuracy.

        _pre_forward: It tells that how to use a specific nn.functional
                      when input, weight and bias are given. This
                      method must be implemented on your own.

        forward: Achieve the flowing of graident in BNN via
                 reparameterization trick and local reparameterization
                 trick. Similar to the `kl_loss`, one have to repeatly
                 forwarding to achieve sufficient accuracy of both
                 loss and gradient. Refer to references for details.

    Attributes:
        prior: The pdf of prior and have to be manually specified.
            Notice that the function should support the flowing
            of gradient.

    References:
        [1] https://arxiv.org/pdf/1505.05424.pdf
        [2] https://arxiv.org/pdf/1506.02557.pdf
    '''
    def __init__(self, prior_sig):
        super(normalModule, self).__init__()
        self.prior_sig = prior_sig

    ''' the initialization strategy follows
            https://github.com/xuanqing94/BayesianDefense
    '''
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
        self.weight_rho.data.fill_(np.log(0.1))
        # self.weight_rho.data.normal_(np.log(0.1), 0.1)
        if self.bias_mu is not None:
            fan_in, _ =nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            self.bias_rho.data.fill_(np.log(0.1))
            # self.bias_rho.data.normal_(np.log(0.1), 0.1)

    def kl_loss(self):
        res = (self.weight_rho.exp() ** 2 + self.weight_mu ** 2).sum() / (2 * self.prior_sig**2) - self.weight_rho.sum()

        if self.bias_mu is not None:
            res += (self.bias_rho.exp() ** 2 + self.bias_mu ** 2).sum() / (2 * self.prior_sig**2) - self.bias_rho.sum()

        return res

    def _pre_forward(self, x, weight, bias):
        raise NotImplementedError

    def forward(self, x):
        out_mu = self._pre_forward(x, self.weight_mu, self.bias_mu)

        ''' calculate post-var '''
        weight_var = self.weight_rho.exp() ** 2
        if self.bias_mu is not None:
            bias_var = self.bias_rho.exp() ** 2
        else:
            bias_var = None
        out_var = self._pre_forward(x**2, weight_var, bias_var)

        ''' post-sampling, local reparameterization tricks '''
        return out_mu + out_var.sqrt() * out_var.data.clone().normal_(0, 1)


class normalLinear(normalModule):
    ''' The usage is almost the same as `torch.nn.Linear` in PyTorch 1.6.

    References:
        [1] https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L34
        [2] https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
    '''
    def __init__(self, prior, in_features, out_features, bias=True):
        super(normalLinear, self).__init__(prior)
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = Parameter(
            torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(
            torch.Tensor(out_features, in_features))

        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def _pre_forward(self, x, weight, bias):
        return F.linear(x, weight, bias)


class normalConv2d(normalModule):
    ''' Only option `padding_mode` is not supported, and the rest 
    is almost the same as `torch.nn.Conv2d` in PyTorch 1.6.

    References:
        [1] https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L260
        [2] https://pytorch.org/docs/1.6.0/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
    '''
    def __init__(self, prior,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(normalConv2d, self).__init__(prior)

        def _pair(x):
            if isinstance(x, container_abcs.Iterable):
                return x
            return tuple(repeat(x, 2))

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight_mu = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        self.weight_rho = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def _pre_forward(self, x, weight, bias):
        return F.conv2d(x, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
