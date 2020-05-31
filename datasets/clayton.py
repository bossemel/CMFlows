import matplotlib.pyplot as plt
import numpy as np
import datasets
import datasets.util
from sklearn import model_selection
import sys


class CLAYTON:
    class Data:
        def __init__(self, data, obs=3000):

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


def sample_data(obs, tau=0.9):

    # @Todo: Random Seed
    theta = 2 * tau / (1 - tau)

    # CLAYTON copula
    U = np.random.uniform(size=obs)
    W = np.random.uniform(size=obs)

    if theta <= -1:
        raise ValueError('the parameter for clayton copula should be more than -1')
    elif theta == 0:
        raise ValueError('The parameter for clayton copula should not be 0')

    if theta < sys.float_info.epsilon:
        V = W
    else:
        V = U * (W**(-theta / (1 + theta)) - 1 + U**theta)**(-1 / theta)

    # Todo: Random Seed
    xx = np.concatenate([U.reshape(-1, 1), V.reshape(-1, 1)], axis=1)
    train, testval = model_selection.train_test_split(xx, test_size=0.2)
    val, test = model_selection.train_test_split(testval, test_size=0.5)
    return train, val, test
