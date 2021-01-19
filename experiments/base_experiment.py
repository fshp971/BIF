import pickle
import logging
import numpy as np
import torch

import models
import utils


logger = logging.getLogger()


class BaseExperiment():
    def __init__(self, args):
        self.save_dir = args.save_dir
        self.burn_in_steps = args.burn_in_steps
        self.eval_freq = args.eval_freq
        self.cpu = args.cpu
        self.batch_size = args.batch_size
        self.is_vi = args.is_vi
        self.samp_num = args.mcmc_samp_num
        self.exp_type = args.exp_type
        self.resume_path = args.resume_path

        if args.save_name is None:
            self.save_name = args.exp_type
        else:
            self.save_name = args.save_name

        self.log = dict()
        self.log['user_time'] = 0

        ''' load dataset '''
        dataset_list = utils.get_dataset(args.dataset)
        self.trainset = dataset_list[0]
        if len(dataset_list) == 2:
            self.testset = dataset_list[1]

        ''' burn_in_sampler is provided only for `burn_in_stage` '''
        self.burn_in_sampler = utils.DataSampler(
            self.trainset, batch_size=args.batch_size)

        ''' train_sampler is provided only for `forget_stage` '''
        self.train_sampler = utils.DataSampler(
            self.trainset, batch_size=args.ifs_iter_bs)

        ''' train_loader is provided for evaluation '''
        self.train_loader = utils.DataLoader(self.trainset, self.batch_size)

        self.ifs_params = {
            'scaling': args.ifs_scaling,
            'iter_T': args.ifs_iter_T,
            'samp_T': args.ifs_samp_T,
            'iter_bs': args.ifs_iter_bs,
            'rm_bs': args.ifs_rm_bs,
            'kill_num': args.ifs_kill_num,
        }

        ''' generate the indices of forgetted data '''
        self.forgetted_train_idx = self.generate_forget_idx(
            self.trainset, self.ifs_params['kill_num'],)
        ''' forgetting follows random order '''
        self.forgetted_train_idx = np.random.permutation(self.forgetted_train_idx)

        utils.add_log(self.log, 'forgetted_train_idx', self.forgetted_train_idx)

        ''' build a loader on forgetted data (for evaluation) '''
        self.forgetted_train_loader = utils.DataLoader(self.trainset,
            batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.forgetted_train_loader.set_sampler_indices(self.forgetted_train_idx)

        self.optim_params = {
            'opt': args.optim,
            'lr': args.lr / len(self.trainset),
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'sghmc_alpha': args.sghmc_alpha,
        }

        self.lr_params = {
            'lr': args.lr / len(self.trainset),
            'lr_decay_exp': args.lr_decay_exp,
            'lr_decay_rate': args.lr_decay_rate,
            'lr_decay_freq': args.lr_decay_freq,
        }

    def run(self):
        real_save_name = '{}-ckpt'.format(self.save_name)

        if self.exp_type == 'burn-in-remain':
            self.burn_in_sampler.remove( self.forgetted_train_idx )
            self.train_loader.remove( self.forgetted_train_idx )
            self.burn_in_stage()
            self.save_checkpoint(real_save_name)

        elif self.exp_type == 'forget':
            state_dict = torch.load(self.resume_path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

            self.forget_stage()
            self.save_checkpoint(real_save_name)

        else:
            raise ValueError('exp {} not found'.format(self.exp_type))

    def burn_in_stage(self):
        raise NotImplementedError

    def forget_stage(self):
        ''' step 1, obtain the indices of forgetted data '''
        self.forgetted_train_idx_loader = utils.IndexBatchSampler(
            batch_size=self.ifs_params['rm_bs'],
            indices=self.forgetted_train_idx)

        ''' step 2, obtain train loader (for evaluation) '''
        self.train_loader.remove(self.forgetted_train_idx)

        ''' forgetting preprocessing end '''

    def eval_one_epoch(self, model, loader):
        raise NotImplementedError

    def generate_forget_idx(self, dataset, kill_num):
        raise NotImplementedError

    def save_checkpoint(self, name, save_log=True, save_model=True):
        save_dir = self.save_dir
        log = self.log
        model = self.model
        optimizer = self.optimizer

        if save_log:
            with open('{}/{}-log.pkl'.format(save_dir, name), 'wb') as f:
                pickle.dump(log, f)

        if save_model:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, '{}/{}-model.pkl'.format(save_dir, name))
