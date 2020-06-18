import datasets.copulas
from utils.split_train_test import split_train_val_test
import numpy as np
import scipy.stats


def marginal_transform(inputs, marginal):
    if marginal == 'GAUSSIAN':
        norm = scipy.stats.norm()
        inputs = norm.ppf(inputs)
    elif marginal == 'UNIFORM':
        return inputs
    elif marginal == 'LOGNORM':
        lognorm = scipy.stats.lognorm()
        inputs = lognorm.ppf(inputs)
    elif marginal == 'GAMMMA':
        gamma = scipy.stats.gamma()
        inputs = gamma.ppf(gamma)
    else:
        raise NotImplementedError
    return inputs


class Joint_Distr():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):
        self.xx = Joint_Distr.sampler(self, args)
        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def sampler(self, args, obs=None):

        copula_xx = Copula_Joint.sampler(self, args, transform=False)

        marginal_1 = marginal_transform(copula_xx[:, 0], args.marginal_1)
        marginal_2 = marginal_transform(copula_xx[:, 1], args.marginal_2)

        xx = np.concatenate([marginal_1.reshape(-1, 1), marginal_2.reshape(-1, 1)], axis=1)
        return xx


class Copula_Joint():
    """Class for samples from a copula.
    """
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):

        self.xx = Copula_Joint.sampler(self, args)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def sampler(self, args, transform=True):
        copula_sampler = datasets.copulas.Copula_sampler(args, transform=False)
        copula_xx = copula_sampler.xx

        if transform:
            if self.transform_fct == 'GAUSSIAN':
                norm = scipy.stats.norm()
                x_1 = norm.ppf(copula_xx[:, 0])
                x_2 = norm.ppf(copula_xx[:, 1])
                copula_xx = np.concatenate([x_1.reshape(-1, 1), x_2.reshape(-1, 1)], axis=1)

            else:
                raise NotImplementedError

        return copula_xx


class Marginals():
    """Class for univariate samples
    """
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):
        self.xx = Marginals.sampler(self, args)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = args.obs

    def sampler(self, args, obs=None):

        if args.marginal == 'GAUSSIAN':
            assert args.mu is not None, 'Please specify mean mu for %r distribution' % (args.marginal)
            assert args.var is not None, 'Please specify variance var for %r distribution' % (args.marginal)
            dataset = np.random.normal(loc=args.mu,
                                       scale=args.var,
                                       size=[args.obs if obs is None else obs])
        elif args.marginal == 'UNIFORM':
            assert hasattr(args, 'low'), 'Please specify lower bound a for %r distribution' % (args.marginal)
            assert hasattr(args, 'high'), 'Please specify upper bound b for %r distribution' % (args.marginal)

            dataset = np.random.uniform(low=args.low,
                                        high=args.high,
                                        size=[args.obs if obs is None else obs])
        elif args.marginal == 'GAMMA':
            assert args.a_param is not None, 'Please specify a_param for %r distribution' % (args.marginal)

            dataset = scipy.stats.gamma.rvs(a=5, size=[args.obs if obs is None else obs])

        return dataset.reshape(-1, 1)
