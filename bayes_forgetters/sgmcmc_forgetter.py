from .bif_forgetter import bifForgetter


class sgmcmcForgetter(bifForgetter):
    ''' bifForgetter for SGMCMC

        Args:
            optimizer (torch.optim): sgmcmc optimizer that used to
                sample (update) model parameter (i.e., params) for
                estimating the expectation terms in I(target).

            samp_T (int): the number of times of SGMCMC sampling for
                estimating the expectation terms in I(target).
    '''
    def __init__(self, model, optimizer, params, cpu, iter_T, scaling, samp_T):
        super(sgmcmcForgetter, self).__init__(
            model, params, cpu, iter_T, scaling)
        self.optimizer = optimizer
        self.param_dict['samp_T'] = samp_T

    def _apply_sample(self, z):
        ''' Conducts SGMCMC sampling for one step (for estimating the
        expectation terms in I(target)).

        Args:
            z (torch.Tensor): a batch of datums sampled from the current
                remaining set.
        '''
        raise NotImplementedError('you should implement `sgmcmcForgetter._apply_sample` yourself before using it')

    def forget(self, target, sampler):
        ''' main function to conduct BIF

        Args:
            target (torch.Tensor): the to be removed datums.

            sampler (iterable): iterable that can repeatly sample
                a batch of datums from the current remaining set.
        '''

        ''' first, estimates the expectation of gradient G_theta(target)
        via Monte Carlo method
        '''
        v_grad = None
        for i in range( self.param_dict['samp_T'] ):
            self._apply_sample( next(sampler) )
            tmp = self._z_fun_grad(target)
            if v_grad is None: v_grad = tmp
            else:
                for vg, pg in zip(v_grad, tmp): vg += pg
        for vg in v_grad: vg /= self.param_dict['samp_T']

        ''' next, repeatedly compute H_theta(S)^(-1) @ G,
        formula: hh_{t+1} = v_grad + (I - H(zz_t)) @ hh_t
                          = v_grad + hh_t - H(zz_t) @ hh_t
        '''
        hh = [vg.clone() for vg in v_grad]
        for tt in range( self.param_dict['iter_T'] ):
            ''' estimates the expectation of hvp H_theta @ G '''
            tmp = None
            for ii in range( self.param_dict['samp_T'] ):
                ''' estimates the expectation via Monte Carlo method '''
                self._apply_sample( next(sampler) )
                zz = next(sampler)
                ''' hessian-vector product '''
                rep_tmp = self._hvp(hh, zz)
                if tmp is None: tmp = rep_tmp
                else:
                    for tp, rtp in zip(tmp, rep_tmp):
                        tp += rtp
            for tp in tmp: tp /= self.param_dict['samp_T']

            for hg, vg, pg in zip(hh, v_grad, tmp):
                hg += vg - pg

        target_grad = hh

        ''' re-scaling: (scaling * H^(-1))v  =>  H^(-1)v '''
        for tg in target_grad:
            tg *= - self.param_dict['scaling']

        ''' apply the forgetting-gradient '''
        for pp, tg in zip(self.params, target_grad):
            pp.data -= tg
