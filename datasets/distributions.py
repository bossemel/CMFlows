import torch
from scipy import stats
import datasets.copulas
from utils.split_train_test import split_train_val_test
import numpy as np
import scipy.stats
from scipy.stats import gamma


def sample_normal_uniform(obs):
    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5],
                                                         [0.5, 1.]])
    # Generate random samples from multivariate normal with correlation .5
    x = mvnorm.rvs(obs)

    norm = stats.norm()
    x_unif = norm.cdf(x)
    return(x_unif)


class Gaussian():

    def __init__(self, noise=0.5):
        self.noise = noise

    def sampler(self, obs):
        return torch.from_numpy(sample_normal_uniform(obs).astype('float32'))


def copula_corr_joint(cop_type='CLAYTON', marginal='GAUSSIAN', obs=1000, tau=0.5, df=2, seed=5):
    xx = datasets.sample_copulas(cop_type, obs, tau, df, seed)
    if marginal == 'GAUSSIAN':
        invnorm = stats.invgauss(mu=0)
        x_1 = invnorm.cdf(xx[:, 0])
        x_2 = invnorm.cdf(xx[:, 1])
    else:
        raise NotImplementedError
    return x_1, x_2


class Copula_Joint():
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):

        if args.tau:
            self.tau = args.tau
        if args.theta:
            self.theta = args.theta
        if args.df:
            self.df = args.df
        self.obs = args.obs
        self.transform_fct = args.transform_fct
        self.cop_type = args.copula
        self.marginal = args.marginal
        self.seed = args.random_seed
        self.obs = args.obs

        args.dataset = args.copula
        Copula_Joint.copula_corr_joint(self, args)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def copula_corr_joint(self, args):
        copula_sampler = datasets.copulas.Copula_sampler(args, transform=False)
        copula_xx = copula_sampler.xx

        if self.marginal == 'GAUSSIAN':
            norm = scipy.stats.norm()
            x_1 = norm.ppf(copula_xx[:, 0])
            x_2 = norm.ppf(copula_xx[:, 1])
        else:
            raise NotImplementedError

        xx = np.concatenate([x_1.reshape(-1, 1), x_2.reshape(-1, 1)], axis=1)
        self.xx = xx


class Marginals():
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):

        self.marginal = args.marginal
        self.seed = args.random_seed
        self.obs = args.obs

        Marginals.sampler(self, args)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = args.obs

    def sampler(self, args, obs=None):

        if self.marginal == 'GAUSSIAN':
            assert args.mu is not None, 'Please specify mean mu for %r distribution' % (self.marginal)
            assert args.var is not None, 'Please specify variance var for %r distribution' % (self.marginal)
            dataset = np.random.normal(loc=args.mu,
                                       scale=args.var,
                                       size=[args.obs if obs is None else obs])
        elif self.marginal == 'UNIFORM':
            assert hasattr(args, 'low'), 'Please specify lower bound a for %r distribution' % (self.marginal)
            assert hasattr(args, 'high'), 'Please specify upper bound b for %r distribution' % (self.marginal)

            dataset = np.random.uniform(low=args.low,
                                        high=args.high,
                                        size=[args.obs if obs is None else obs])
        elif self.marginal == 'GAMMA':
            assert args.a_param is not None, 'Please specify a_param for %r distribution' % (self.marginal)

            dataset = gamma.rvs(a=5, size=10000)

        self.xx = dataset.reshape(-1, 1)
        return self.xx
