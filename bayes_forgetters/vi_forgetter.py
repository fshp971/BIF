from .bif_forgetter import bifForgetter


class viForgetter(bifForgetter):
    def __init__(self, model, params, cpu, iter_T, scaling):
        super(viForgetter, self).__init__(
            model, params, cpu, iter_T, scaling)


class vbnnForgetter(bifForgetter):
    ''' variational Bayesian neural network forgetter

        Args:
            model (torch.nn.Module): the variational neural network.

            bbp_T (int): the number of repeated backpropagation when
                performing "Bayes by Backprop" (BBP) [1].

        References:
            [1] https://arxiv.org/pdf/1505.05424.pdf
    '''
    def __init__(self, model, params, cpu, iter_T, scaling, bbp_T):
        super(vbnnForgetter, self).__init__(
            model, params, cpu, iter_T, scaling)
        self.param_dict['bbp_T'] = bbp_T

    def _z_fun_grad(self, z):
        res = None
        for i in range( self.param_dict['bbp_T'] ):
            tmp = super(vbnnForgetter, self)._z_fun_grad(z)
            if res is None: res = tmp
            else:
                for rp, gp in zip(res, tmp): rp += gp
        for rp in res: rp /= self.param_dict['bbp_T']
        return res

    def _hvp(self, hh, zz):
        res = None
        for i in range( self.param_dict['bbp_T'] ):
            tmp = super(vbnnForgetter, self)._hvp(hh, zz)
            if res is None: res = tmp
            else:
                for rp, gp in zip(res, tmp): rp += gp
        for rp in res: rp /= self.param_dict['bbp_T']
        return res
