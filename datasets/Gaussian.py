import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from scipy import stats
import datasets
import datasets.util


class GAUSSIAN:
    class Data:
        def __init__(self, data, obs=3000):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]
            self.obs = obs

    def __init__(self, obs=3000):

        trn, val, tst = sample_data(obs)
        print(trn.shape, val.shape, tst.shape)

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
    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.9],
                                                         [0.9, 1.]])
    # Generate random samples from multivariate normal with correlation .5
    x = mvnorm.rvs(obs)

    norm = stats.norm()
    x_unif = norm.cdf(x)
    train, testval = model_selection.train_test_split(x_unif, test_size=0.2)
    val, test = model_selection.train_test_split(testval, test_size=0.5)
    return train, val, test
