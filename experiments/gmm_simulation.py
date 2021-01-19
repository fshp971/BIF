from datetime import datetime
import pickle
import logging
import numpy as np
import torch

import bayes_forgetters
import models
import utils
from .base_experiment import BaseExperiment


logger = logging.getLogger()


class GMMForgetter(bayes_forgetters.viForgetter):
    def _fun(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        self.model.train()
        return self.model.loss(x, self.model(x))

    def _z_fun(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        self.model.train()
        return self.model.x_loss(x, self.model(x))


class mcmcForgetter(bayes_forgetters.sgmcmcForgetter):
    def _apply_sample(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        lo = self.model.loss(x, self.model(x))
        self.optimizer.zero_grad()
        lo.backward()
        self.optimizer.step()

    def _fun(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        self.model.train()
        return self.model.loss(x, self.model(x))

    def _z_fun(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        self.model.train()
        return self.model.x_loss(x, self.model(x))


class GMMSimulation(BaseExperiment):
    def __init__(self, args):
        super(GMMSimulation, self).__init__(args)

        ''' get arch '''
        self.arch_params = {
            'kk': args.gmm_kk,
            'dim': 2,
            'std': [args.gmm_std, args.gmm_std],
            'n': None,
        }
        if self.is_vi:
            self.model = models.VariationalGMM(**self.arch_params)
        else:
            self.model = models.GMM(**self.arch_params)

        self.optimizer = utils.get_optim(
            self.model.parameters(), **self.optim_params)

    def run(self):
        super(GMMSimulation, self).run()

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

            loss, log_pp = self.burn_in_one_step(z)

            torch.cuda.synchronize()
            end_time = datetime.now()
            user_time = (end_time - start_time).total_seconds()
            ''' end calculation of time '''

            self.log['user_time'] += user_time

            utils.add_log(self.log, 'burn_in_loss', loss)
            utils.add_log(self.log, 'burn_in_log_pp', log_pp)

            if (step+1) % self.eval_freq == 0:
                logger.info('burn-in step [{}/{}]:'
                            .format(step+1, self.burn_in_steps))
                logger.info('user time {:.3f} sec \t'
                            'cumulated user time {:.3f} mins'
                            .format(user_time, self.log['user_time']/60) )
                logger.info('burn-in loss {:.2e} \t'
                            'burn-in log_pp {:.2e}'
                            .format(loss, log_pp) )

                fo_train_loss, fo_train_log_pp = self.evaluate(self.model, self.forgetted_train_loader)
                logger.info('forgetted train loss {:.2e} \t'
                            'train log pp {:.2e}'
                            .format(fo_train_loss, fo_train_log_pp) )

                logger.info('')

    def forget_stage(self):
        super(GMMSimulation, self).forget_stage()
        forgetted_train_idx_loader = self.forgetted_train_idx_loader
        train_loader = self.train_loader
        forgetted_train_loader = self.forgetted_train_loader
        train_sampler = self.train_sampler

        ''' initialize remaining set size '''
        self.model.n = len(train_sampler)

        if self.is_vi:
            forgetter = GMMForgetter(
                model = self.model,
                params = self.model.parameters(),
                cpu = self.cpu,
                iter_T = self.ifs_params['iter_T'],
                scaling = self.ifs_params['scaling'],)
        else:
            forgetter = mcmcForgetter(
                model = self.model,
                optimizer = self.optimizer,
                params = self.model.parameters(),
                cpu = self.cpu,
                iter_T = self.ifs_params['iter_T'],
                scaling = self.ifs_params['scaling'],
                samp_T = self.ifs_params['samp_T'], )

        self.forget_eval_one_time(train_loader, forgetted_train_loader)
        logger.info('')

        for ii in forgetted_train_idx_loader:
            xx, yy = self.trainset[ii]
            xx, yy = torch.Tensor(xx), torch.Tensor(yy)

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

            self.forget_eval_one_time(train_loader, forgetted_train_loader)
            logger.info('')

    def burn_in_one_step(self, z):
        model = self.model
        optimizer = self.optimizer
        is_vi = self.is_vi
        cpu = self.cpu

        model.train()
        x, _ = z

        if not cpu: x = x.cuda()

        _y = model(x)
        loss = model.loss(x, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_pp = _y.max(dim=1)[0].data.log().mean()

        return loss.item(), log_pp.item()

    def mcmc_sampling_points(self, samp_num, sampler):
        assert not self.is_vi
        model = self.model
        optimizer = self.optimizer
        cpu = self.cpu
        points = []

        model.train()
        for i in range(samp_num):
            x, _ = next(sampler)
            if not cpu: x = x.cuda()
            _y = model(x)
            loss = model.loss(x, model(x))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            points.append( model.dump_numpy_dict()['mk'].copy() )

        points = np.array(points)

        return points

    def forget_eval_one_time(self, train_loader, forgetted_train_loader):
        remain_train_loss, remain_train_log_pp = self.evaluate(self.model, train_loader)
        forgetted_train_loss, forgetted_train_log_pp = self.evaluate(self.model, forgetted_train_loader)
        utils.add_log(self.log, 'remain_train_loss', remain_train_loss)
        utils.add_log(self.log, 'remain_train_log_pp', remain_train_log_pp)
        utils.add_log(self.log, 'forgetted_train_loss', forgetted_train_loss)
        utils.add_log(self.log, 'forgetted_train_log_pp', forgetted_train_log_pp)
        logger.info('remaining train loss {:.2e} \t train log pp {:.2e}'
                    .format(remain_train_loss, remain_train_log_pp))
        logger.info('forgetted train loss {:.2e} \t train log pp {:.2e}'
                    .format(forgetted_train_loss, forgetted_train_log_pp))

    def evaluate(self, model, loader):
        cpu = self.cpu
        is_vi = self.is_vi

        loss = utils.AverageMeter()
        ''' log_pp stands for log predictive probability '''
        log_pp = utils.AverageMeter()

        model.eval()
        for x, _ in loader:
            if not cpu: x = x.cuda()

            with torch.no_grad():
                _y = model(x)
                lo = model.loss(x, _y)

            loss.update(lo.item(), len(x))
            log_pp.update(_y.max(dim=1)[0].data.log().mean().item(), len(x))

        return loss.average(), log_pp.average()

    def generate_forget_idx(self, dataset, kill_num):
        ''' for special case '''
        randidx = []
        hf = kill_num // 2

        idx = np.where(dataset.y==0)[0]
        sorted_idx = dataset.x[:,0][idx].argsort()
        randidx.append( idx[sorted_idx[len(sorted_idx)-hf:]] )

        idx = np.where(dataset.y==2)[0]
        sorted_idx = dataset.x[:,1][idx].argsort()
        # randidx.append( idx[sorted_idx[len(sorted_idx)-(kill_num-hf):]] )
        randidx.append( idx[sorted_idx[:(kill_num-hf)]] )

        randidx = np.concatenate(randidx)
        randidx = np.random.permutation(randidx)
        return randidx

    def save_checkpoint(self, name, save_log=True, save_cluster=True, save_model=True, save_samp_pts=True):
        save_dir = self.save_dir
        model = self.model

        if save_samp_pts and (self.exp_type!='forget') and (not self.is_vi):
            points = self.mcmc_sampling_points(self.samp_num, self.burn_in_sampler)
            with open('{}/{}-samp-pts.pkl'.format(save_dir,name),'wb') as f:
                pickle.dump(points, f)

        super(GMMSimulation, self).save_checkpoint(name, save_log, save_model)

        if save_cluster:
            with open('{}/{}-cluster.pkl'.format(save_dir,name), 'wb') as f:
                pickle.dump(model.dump_numpy_dict(), f)
