import torch


class bifForgetter():
    ''' Implementation of Bayesian inference forgetting (BIF) forgetter.

    It will first calculate the following function I(target):

        I(target) = -H_theta^(-1) @ G_theta

    and then remove the influence from model parameter as follow:

        theta_new = theta + (-1) * I(target)

    where theta is the model parameter, H_theta is the hessian matrix
    of F(theta, S) (see the energy function in BIF), and G_theta is the
    gradient of h(theta, z) (see the energy function in BIF).

    The calculation of I(target) relies on Taylor series (for calculating
    hessian-inverse) and efficient hessian-vector product, just as the
    following references suggest.

    References:
        [1] https://arxiv.org/pdf/1602.03943.pdf
        [2] https://arxiv.org/pdf/1703.04730.pdf

    Args:
        model (torch.nn.Module): the Bayesian model.

        params (iterable): iterable of parameters to optimize.

        cpu (bool): True=use cpu, False=use cuda.

        iter_T (int): the number of iterations that computing
            a single hessian-inverse

        scaling (float): The algorithm for calculating hessian-inverse
            converges iff |det(H)| <= 1, and `scaling` is provided to
            let hessian matrix H meet that condition. Specifically, in
            each iteration, the following inverse matrix will be calculated,
                (scaling * H)^(-1) = (1/scaling) * H^(-1),
            and once the whole iterations finished, the obtained result
            will be rescaled to its true value.
    '''
    def __init__(self, model, params, cpu, iter_T, scaling):
        self.model = model
        self.params = [pp for pp in params]
        self.cpu = cpu
        self.param_dict = dict(iter_T=iter_T, scaling=scaling)

    def _fun(self, z):
        ''' Calculates the empirical energy function F(theta, S) in BIF.

        Args:
            z (torch.Tensor): a batch of datums sampled from the
                current remaining set.
        '''
        raise NotImplementedError('you should implement `bifForgetter._fun` yourself before using it')

    def _z_fun(self, z):
        ''' Calculates the function h(theta, z) in BIF.

        Args:
            z (torch.Tensor): datums that going to be removed.
        '''
        raise NotImplementedError('you should manually implement `bifForgetter._z_fun` yourself before using it')

    def _z_fun_grad(self, z):
        ''' Calculates the gradient of h(theta, z) in BIF.

        Args:
            z (torch.Tensor): datums that going to be removed.
        '''
        z_lo = self._z_fun(z)
        return torch.autograd.grad(z_lo, self.params)

    def _hvp(self, hh, zz):
        ''' Calculates the hessian-vector product.
        Given: `hh`, `zz`; return: (scaling * H(zz)) @ hh.

        Args:
            hh (torch.Tensor): the intermediate result (a vector) when
                iteratively computing inverse-hessian-vector product.

            zz (torch.tensor): a batch of datums sampled from
                the current remaining set.
        '''
        lo = self._fun(zz)
        lo *= self.param_dict['scaling']
        tmp = torch.autograd.grad(lo, self.params, create_graph=True)

        lloo = 0
        for hg, pg in zip(hh, tmp):
            lloo += (hg*pg).sum()
        tmp = torch.autograd.grad(lloo, self.params)

        return tmp

    def forget(self, target, sampler):
        ''' main function to conduct BIF

        Args:
            target (torch.Tensor): the to be removed datums.

            sampler (iterable): iterable that can repeatedly sample
                a batch of datums from the current remaining set.
        '''

        ''' first compute G_theta(target) '''
        v_grad = self._z_fun_grad(target)

        ''' next repeatedly compute H_theta(S)^(-1) @ G,
        formula: hh_{t+1} = v_grad + (I - H(zz_t)) @ hh_t
                          = v_grad + hh_t - H(zz_t) @ hh_t
        '''
        hh = [vg.clone() for vg in v_grad]
        for ii in range( self.param_dict['iter_T'] ):
            zz = next(sampler)
            ''' hessian-vector product '''
            tmp = self._hvp(hh, zz)
            for hg, vg, pg in zip(hh, v_grad, tmp):
                hg += vg - pg

        target_grad = hh

        assert target_grad is not None

        ''' re-scaling: (scaling * H^(-1))v  =>  H^(-1)v '''
        for tg in target_grad:
            tg *= - self.param_dict['scaling']

        ''' apply the forgetting-gradient '''
        for pp, tg in zip(self.params, target_grad):
            pp.data -= tg
