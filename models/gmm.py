import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class VariationalGMM(nn.Module):
    def __init__(self, kk, dim, std, n):
        super(VariationalGMM, self).__init__()
        self.kk = kk
        self.dim = dim
        self.n = n
        assert len(std) == dim
        self.std = torch.Tensor(std)
        self.mk = Parameter(torch.Tensor(dim, kk))
        ''' sk = log1p( exp(rhok) ) '''
        self.rhok = Parameter(torch.Tensor(dim, kk))

        self.mk.data.normal_(0, 1)
        self.rhok.data.normal_(-3, 0.1).abs_()

    def forward(self, x):
        mk2 = (self.mk**2).sum(dim=0, keepdim=True)
        sk2 = (self.rhok.exp().log1p()**2).sum(dim=0, keepdim=True)

        x = x.reshape(-1, self.dim)
        out = x.mm(self.mk) - 0.5 * (mk2 + sk2)
        out = F.softmax(out, dim=1)
        return out

    def pa_loss(self):
        ''' i.e. kl-loss '''
        mk2 = (self.mk**2)
        sk = self.rhok.exp().log1p()
        sk2 = sk**2

        out = ((mk2.sum(dim=1) + sk2.sum(dim=1)) / (2*self.std**2)).sum() - sk.log().sum()
        out += self.kk * (0.5 * self.dim - self.std.log().sum())
        return out

    def x_loss(self, x, _y):
        mk2 = (self.mk**2).sum(dim=0, keepdim=True)
        sk2 = (self.rhok.exp().log1p()**2).sum(dim=0, keepdim=True)
        x = x.reshape(-1, self.dim)
        real_n = len(x)

        out = (-2*x.mm(self.mk) + mk2 + sk2) * 0.5 + _y.clamp(min=1e-20).log()
        out = (out * _y).sum()
        out += (x**2).sum() * 0.5
        out += self.dim * real_n * ( np.log(self.kk) + np.log(np.sqrt(2*np.pi)) )
        return out

    def loss(self, x, _y):
        return self.pa_loss() + self.x_loss(x, _y) * (self.n/len(x))

    def dump_numpy_dict(self):
        return {'kk': self.kk,
                'dim': self.dim,
                'n': self.n,
                'std': np.array(self.std.data.cpu()),
                'mk': np.array(self.mk.data.cpu()),
                'sk': np.array(self.rhok.data.exp().log1p().cpu()), }


class GMM(nn.Module):
    def __init__(self, kk, dim, std, n):
        super(GMM, self).__init__()
        self.kk = kk
        self.dim = dim
        self.n = n
        assert len(std) == dim
        self.std = torch.Tensor(std)
        self.mk = Parameter(torch.Tensor(kk, dim))

        self.mk.data.normal_(0, 1)

    def forward(self, x):
        x = x.reshape(-1, 1, self.dim)
        out = ((x - self.mk)**2) * (-0.5) - np.log(np.sqrt(2*np.pi))
        out = out.sum(dim=2)
        return out.exp()

    def pa_loss(self):
        ''' i.e. log_prior '''
        out = (self.mk**2)*(-0.5) -self.std.log() -np.log(np.sqrt(2*np.pi))
        return -out.sum()

    def x_loss(self, x, _y):
        out = _y.mean(dim=1).log().sum()
        return -out

    def loss(self, x, _y):
        return self.pa_loss() + self.x_loss(x, _y) * (self.n/len(x))

    def dump_numpy_dict(self):
        return {'kk': self.kk,
                'dim': self.dim,
                'std': np.array(self.std.data.cpu()),
                'mk': np.array(self.mk.data.t().cpu())}
