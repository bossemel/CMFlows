from sklearn.datasets import make_swiss_roll
import torch
from scipy import stats
from datasets.copulas import sample_copulas


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
    xx = sample_copulas(cop_type, obs, tau, df, seed)
    if marginal == 'GAUSSIAN':
        invnorm = stats.invgauss(mu=0)
        x_1 = invnorm.cdf(xx[:, 0])
        x_2 = invnorm.cdf(xx[:, 1])
    else:
        raise NotImplementedError
    return x_1, x_2


class Copula_Joint():

    def __init__(self, cop_type='CLAYTON', marginal='GAUSSIAN', tau=0.5, df=2, seed=5, noise=0.5):
        # @Todo: Look at what noise is for
        self.noise = noise
        self.cop_type = cop_type
        self.marginal = marginal
        self.tau = tau
        self.df = df
        self.seed = seed

    def sampler(self, obs):
        # @Todo: implement calling two-ddsfs
        samples = copula_corr_joint(self.cop_type, self.marginal, obs, self.tau, self.df, self.seed)
        # print(samples[0].shape)
        return torch.from_numpy(samples[0].astype('float32')).reshape(-1,1)
