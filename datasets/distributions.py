from sklearn.datasets import make_swiss_roll
import torch
from scipy import stats
import datasets  # .copulas import sample_copulas
from utils.split_train_test import split_train_val_test
import numpy as np
import scipy.stats


class Distr(object):

    hasenergyf = False
    hassplr = False

    def energy(self, x):
        raise NotImplementedError

    def sampler(self, x):
        raise NotImplementedError


class SwissRoll(Distr):

    hasenergyf = False
    hassplr = True

    def __init__(self, noise=0.5):
        self.noise = noise

    def sampler(self, n):
        #print(make_swiss_roll(n, self.noise)[0][:, [0, 2]].shape)
        return torch.from_numpy(
            make_swiss_roll(n, self.noise)[0][:, [0, 2]].astype('float32') / 3.)


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
    # print('obs', obs)
    # @Todo: Fix Seed
    xx = datasets.sample_copulas(cop_type, obs, tau, df, seed)
    if marginal == 'GAUSSIAN':
        invnorm = stats.invgauss(mu=0)
        x_1 = invnorm.cdf(xx[:, 0])
        x_2 = invnorm.cdf(xx[:, 1])
    else:
        raise NotImplementedError
    return x_1, x_2


# class Copula_Joint():

#     def __init__(self, cop_type='CLAYTON', marginal='GAUSSIAN', tau=0.5, df=2, seed=5, noise=0.5):
#         # @Todo: Look at what noise is for
#         self.noise = noise
#         self.cop_type = cop_type
#         self.marginal = marginal
#         self.tau = tau
#         self.df = df
#         self.seed = seed

#     def sampler(self, obs):
#         # @Todo: implement calling two-ddsfs
#         samples = copula_corr_joint(self.cop_type, self.marginal, obs, self.tau, self.df, self.seed)
#         # print(samples[0].shape)
#         return torch.from_numpy(samples[0].astype('float32')).reshape(-1,1)


class Copula_Joint:
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
        # @Todo: Fix Seed
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
