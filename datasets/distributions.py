from sklearn.datasets import make_swiss_roll
import torch
from scipy import stats


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
