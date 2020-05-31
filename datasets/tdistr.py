import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from scipy import stats
import datasets
import datasets.util


class TDISTR:
    class Data:
        def __init__(self, data, obs=30000):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]
            self.obs = obs

    def __init__(self, obs=3000):

        trn, val, tst = sample_data(obs)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        datasets.util.plot_hist_marginals(data_split.x)
        plt.show()


def sample_data(obs):
    #Todo: Random Seed
    x = multivariate_t(mu=[0, 0], sigma=[[1., 0.9],
                                         [0.9, 0.5]], dof=2, m=obs)
    # Generate random samples from multivariate normal with correlation .5
    #x = mvt.rvs(obs)

    tt = stats.t(df=2)
    x_unif = tt.cdf(x)
    train, testval = model_selection.train_test_split(x_unif, test_size=0.2)
    val, test = model_selection.train_test_split(testval, test_size=0.5)
    return train, val, test


def multivariate_t(mu, sigma, dof, m):
    """
    Produce m samples of d-dimensional multivariate t distribution

    Args:
        mu (numpy.ndarray): mean vector
        sigma (numpy.ndarray): scale matrix (covariance)
        dof (float): degrees of freedom
        m (int): # of samples to produce

    Returns:
        numpy.ndarray
    """
    d = len(sigma)
    g = np.tile(np.random.gamma(dof / 2, 2 / dof, m), (d, 1)).T
    z = np.random.multivariate_normal(np.zeros(d),sigma,m)
    return mu + z / np.sqrt(g)
