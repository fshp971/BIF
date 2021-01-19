from datetime import datetime
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F

import bayes_forgetters
import models
import utils
from .base_experiment import BaseExperiment


logger = logging.getLogger()


class viForgetter(bayes_forgetters.vbnnForgetter):
    def _fun(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        return self.model.kl_loss() + F.cross_entropy(self.model(x), y) * self.model.n

    def _z_fun(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        return F.cross_entropy(self.model(x), y, reduction='sum')


class mcmcForgetter(bayes_forgetters.sgmcmcForgetter):
    def _apply_sample(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        lo = -self.model.log_prior() + F.cross_entropy(self.model(x), y) * self.model.n
        self.optimizer.zero_grad()
        lo.backward()
        self.optimizer.step()

    def _fun(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        return -self.model.log_prior() + F.cross_entropy(self.model(x), y) * self.model.n

    def _z_fun(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        return F.cross_entropy(self.model(x), y, reduction='sum')


class DeepLearning(BaseExperiment):
    def __init__(self, args):
        super(DeepLearning, self).__init__(args)
        if self.is_vi: self.bbp_T = args.dl_bbp_T
        self.test_loader = utils.DataLoader(self.testset, self.batch_size)

        self.model = self.generate_arch(
            args.arch, args.dataset, args.dl_prior_sig)

        if not self.cpu:
            self.model.cuda()

        self.optimizer = utils.get_optim(
            self.model.parameters(), **self.optim_params)

    def run(self):
        super(DeepLearning, self).run()

    def burn_in_stage(self):
        sampler = self.burn_in_sampler
        optimizer = self.optimizer

        ''' set training set size '''
        self.model.n = len(sampler)

        for step in range(self.burn_in_steps):
            lr = self.lr_params['lr']
            decay = 1.0

            if self.lr_params['lr_decay_exp'] is not None:
                decay = ((step+1) ** self.lr_params['lr_decay_exp'])

            elif self.lr_params['lr_decay_rate'] is not None:
                assert self.lr_params['lr_decay_freq'] is not None
                lr_decay_rate = self.lr_params['lr_decay_rate']
                lr_decay_freq = self.lr_params['lr_decay_freq']
                decay = lr_decay_rate ** (step // lr_decay_freq)

            for group in optimizer.param_groups:
                if 'lr_decay' in group.keys():
                    group['lr_decay'] = decay
                else:
                    group['lr'] = lr * decay

            z = next(sampler)

            ''' start calculation of time '''
            start_time = datetime.now()

            loss, acc = self.burn_in_one_step(z)

            torch.cuda.synchronize()
            end_time = datetime.now()
            user_time = (end_time - start_time).total_seconds()
            ''' end calculation of time '''

            self.log['user_time'] += user_time

            utils.add_log(self.log, 'burn_in_loss', loss)
            utils.add_log(self.log, 'burn_in_acc', acc)

            if (step+1) % self.eval_freq == 0:
                logger.info('burn-in step [{}/{}]:'
                            .format(step+1, self.burn_in_steps))
                logger.info('user time {:.3f} sec \t'
                            'cumulated user time {:.3f} mins'
                            .format(user_time, self.log['user_time']/60) )
                logger.info('burn-in loss {:.2e} \t'
                            'burn-in acc {:.2%}'
                            .format(loss, acc) )

                test_loss, test_acc = self.evaluate(self.model, self.test_loader)
                logger.info('test loss {:.2e} \t'
                            'test acc {:.2%}'
                            .format(test_loss, test_acc) )
                utils.add_log(self.log, 'test_loss', test_loss)
                utils.add_log(self.log, 'test_acc', test_acc)

                train_loss, train_acc = self.evaluate(self.model, self.train_loader)
                logger.info('(maybe remain) train loss {:.2e} \t'
                            'train acc {:.2%}'
                            .format(train_loss, train_acc) )
                utils.add_log(self.log, '(remain) train_loss', train_loss)
                utils.add_log(self.log, '(remain) train_acc', train_acc)

                fo_train_loss, fo_train_acc = self.evaluate(self.model, self.forgetted_train_loader)
                logger.info('forgetted train loss {:.2e} \t'
                            'train acc {:.2%}'
                            .format(fo_train_loss, fo_train_acc) )
                utils.add_log(self.log,'forgetted_train_loss',fo_train_loss)
                utils.add_log(self.log,'forgetted_train_acc',fo_train_acc)

                logger.info('')

    def forget_stage(self):
        super(DeepLearning, self).forget_stage()
        forgetted_train_idx_loader = self.forgetted_train_idx_loader
        train_loader = self.train_loader
        forgetted_train_loader = self.forgetted_train_loader
        train_sampler = self.train_sampler
        test_loader = self.test_loader

        ''' initialize remaining set size '''
        self.model.n = len(train_sampler)

        if self.is_vi:
            forgetter = viForgetter(
                model=self.model,
                params=self.model.parameters(),
                cpu=self.cpu,
                iter_T=self.ifs_params['iter_T'],
                scaling=self.ifs_params['scaling'],
                bbp_T=self.bbp_T,)
        else:
            forgetter = mcmcForgetter(
                model=self.model,
                optimizer=self.optimizer,
                params=self.model.parameters(),
                cpu=self.cpu,
                iter_T=self.ifs_params['iter_T'],
                scaling=self.ifs_params['scaling'],
                samp_T=self.ifs_params['samp_T'],)

        self.forget_eval_one_time(
            train_loader, forgetted_train_loader, test_loader)
        logger.info('')

        for ii in self.forgetted_train_idx_loader:
            ''' create forget-batch '''
            xx, yy = [], []
            for i in ii:
                x, y = self.trainset[i]
                if len(x.shape) == 3: x = x.reshape(1, *x.shape)
                xx.append(x)
                yy.append(y)
            xx, yy = torch.cat(xx), torch.tensor(yy)
            ''' end '''

            scaling = self.ifs_params['scaling'] / len(train_sampler)
            forgetter.param_dict['scaling'] = scaling

            ''' start calculation of time '''
            start_time = datetime.now()

            forgetter.forget([xx,yy], train_sampler)

            torch.cuda.synchronize()
            end_time = datetime.now()
            user_time = (end_time - start_time).total_seconds()
            ''' end calculation of time '''

            self.log['user_time'] += user_time

            train_sampler.remove(ii)

            ''' after removal, update the number of remaining datums '''
            forgetter.model.n = len(train_sampler)

            logger.info('remaining trainset size {}'.format(len(train_sampler)))
            logger.info('user time {:.3f} sec \t'
                        'cumulated user time {:.3f} mins'
                        .format(user_time, self.log['user_time']/60) )

            self.forget_eval_one_time(
                train_loader, forgetted_train_loader, test_loader)
            logger.info('')

    def burn_in_one_step(self, z):
        model = self.model
        optimizer = self.optimizer
        is_vi = self.is_vi
        cpu = self.cpu

        model.train()

        x, y = z
        if not cpu: x, y = x.cuda(), y.cuda()

        if is_vi:
            bbp_T = self.bbp_T
            loss, pred_y = 0, 0

            optimizer.zero_grad()

            for t in range(bbp_T):
                _y = model(x)
                lo = (model.kl_loss() + F.cross_entropy(_y, y) * model.n) / bbp_T
                lo.backward()

                loss += lo.item()
                pred_y += _y.data.softmax(dim=1)

            optimizer.step()

            acc = (pred_y.argmax(dim=1) == y).sum().item() / len(y)

        else:
            _y = model(x)
            lo = - model.log_prior() + F.cross_entropy(_y,y) * model.n

            optimizer.zero_grad()
            lo.backward()
            optimizer.step()

            loss = lo.item()
            acc = (_y.argmax(dim=1) == y).sum().item() / len(y)

        return loss, acc

    def forget_eval_one_time(self,
                             train_loader,
                             forgetted_train_loader,
                             test_loader):
        model = self.model
        remain_train_loss, remain_train_acc = self.evaluate(model, train_loader)
        forgetted_train_loss, forgetted_train_acc = self.evaluate(model, forgetted_train_loader)
        test_loss, test_acc = self.evaluate(model, test_loader)

        utils.add_log(self.log, 'remain_train_loss', remain_train_loss)
        utils.add_log(self.log, 'remain_train_acc', remain_train_acc)
        utils.add_log(self.log,'forgetted_train_loss', forgetted_train_loss)
        utils.add_log(self.log,'forgetted_train_acc', forgetted_train_acc)
        utils.add_log(self.log, 'test_loss', test_loss)
        utils.add_log(self.log, 'test_acc', test_acc)

        logger.info('remaining train loss {:.2e} \t train acc {:.2%}'
                    .format(remain_train_loss, remain_train_acc))
        logger.info('forgetted train loss {:.2e} \t train acc {:.2%}'
                    .format(forgetted_train_loss, forgetted_train_acc))
        logger.info('test loss {:.2e} \t test acc {:.2%}'
                    .format(test_loss, test_acc))

    def evaluate(self, model, loader):
        cpu = self.cpu
        is_vi = self.is_vi

        ''' average log predictive probability '''
        loss = utils.AverageMeter()
        acc = utils.AverageMeter()

        n = len(loader.sampler.indices)

        model.eval()
        for x, y in loader:
            if not cpu: x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                if is_vi:
                    _y = model(x)
                    lo = model.kl_loss() + F.cross_entropy(_y, y) * n
                    lo = lo.item()
                    ac = (_y.argmax(dim=1) == y).sum().item() / len(y)

                else:
                    _y = model(x)
                    lo = - model.log_prior() + F.cross_entropy(_y,y) * n
                    lo = lo.item()
                    ac = (_y.argmax(dim=1) == y).sum().item() / len(y)

            loss.update(lo, len(y))
            acc.update(ac, len(y))

        return loss.average(), acc.average()

    def generate_arch(self, arch, dataset, prior_sig):
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            in_dims = 1

        elif dataset == 'cifar10':
            in_dims = 3

        else:
            raise ValueError('dataset {} is not supported'.format(dataset))

        if self.is_vi:
            if arch == 'mlp':
                return models.normalMLP(prior_sig, in_dims)
            elif arch == 'lenet':
                return models.normalLeNet(prior_sig, in_dims)

        else:
            if arch == 'mlp':
                return models.mcmcMLP(prior_sig, in_dims)
            elif arch == 'lenet':
                return models.mcmcLeNet(prior_sig, in_dims)

        raise ValueError('arch {} is not supported'.format(arch))

    def generate_forget_idx(self, dataset, kill_num):
        kill_val = 0

        if 'targets' in vars(dataset).keys():
            labels = np.array(dataset.targets)
        elif 'labels' in vars(dataset).keys():
            labels = np.array(dataset.labels)
        else:
            raise NotImplementedError

        randidx = np.random.permutation( np.where(labels==kill_val)[0] )
        return randidx[:kill_num]

    def save_checkpoint(self, name, save_log=True, save_model=True):
        super(DeepLearning, self).save_checkpoint(name, save_log, save_model)
