import torch
import torchvision.transforms as transforms

from . import sgmcmc_optim
from . import datasets


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_dataset(dataset):
    if dataset == 'gmm2d':
        return [datasets.GMM2d()]

    ''' below are cv datasets '''
    if dataset == 'mnist' or dataset == 'fashion-mnist':
        normalize = transforms.Normalize((0.5,), (1.,))
        transform_train = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize])
    else:
        normalize = transforms.Normalize((0.5,0.5,0.5,), (1.,1.,1.,))
        transform_train = transforms.Compose(
            [transforms.ToTensor(), normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), normalize])

    if dataset == 'mnist':
        trainset = datasets.MNIST('./data', True, transform_train)
        testset = datasets.MNIST('./data', False, transform_test)
    elif dataset == 'fashion-mnist':
        trainset = datasets.FashionMNIST('./data', True, transform_train)
        testset = datasets.FashionMNIST('./data', False, transform_test)
    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10('./data', True, transform_train)
        testset = datasets.CIFAR10('./data', False, transform_test)
    else:
        raise ValueError('dataset {} is not supported'.format(dataset))

    return [trainset, testset]


def get_optim(parameters, opt, **kwargs):
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']

    if opt == 'sgd':
        momentum = kwargs['momentum']
        return torch.optim.SGD(parameters, momentum=momentum,
                    lr=lr, weight_decay=weight_decay)

    elif opt == 'adam':
        return torch.optim.Adam(parameters,
                    lr=lr, weight_decay=weight_decay)

    elif opt == 'sgld':
        return sgmcmc_optim.SGLD(parameters, lr=lr)

    elif opt == 'sghmc':
        alpha = kwargs['sghmc_alpha']
        return sgmcmc_optim.SGHMC(parameters, lr=lr, alpha=alpha)

    raise ValueError('optim method {} is not supported'.format(opt))


def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate, lr_decay_freq):
    if (lr_decay_rate is None) or (lr_decay_freq is None):
        return

    lr = lr * (lr_decay_rate ** (epoch // lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
