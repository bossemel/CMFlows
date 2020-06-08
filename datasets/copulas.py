import matplotlib.pyplot as plt
import numpy as np
import datasets
import datasets.util
from sklearn import model_selection
import sys
from scipy import stats, special


class Copula_sampler:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):

        self.cop_type = args.dataset
        if args.tau:
            self.tau = args.tau
        if args.theta:
            self.theta = args.theta
        if args.df:
            self.df = args.df
        self.seed = args.seed
        self.obs = args.obs

        Copula_sampler.sample_copulas(self)

        trn, val, tst = split_train_val_test(self.xx, self.seed)

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

    def sample_copulas(self):
        """
        Produce obs samples of 2-dimensional Copula density distribution

        Args:
            cop_type (str): copula type, one of clayton, gumbel, frank
            obs (int): number of samples
            tau (int): tau copula parameter
            seed (int): random seed

        Returns:
            train (numpy.ndarray): training set
            val (numpy.ndarray): validation set
            test (numpy.ndarray): test set
        """
        np.random.seed(self.seed)

        assert self.cop_type in ['CLAYTON', 'FRANK', 'GUMBEL', 'GAUSSIAN', 'TDISTR'], \
            "%r is not a valid copula, choose from %r" % (self.cop_type, ['CLAYTON', 'FRANK', 'GUMBEL', 'GAUSSIAN', 'TDISTR'])

        # Following Copula definitions from
        # https://pydoc.net/copulalib/1.1.0/copulalib.copulalib/
        # Conditional Distribution Method:
        # CLAYTON copula
        if self.cop_type == 'CLAYTON':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.cop_type)
            uu, vv = sample_clayton(self.obs, self.theta, self.seed)

        # FRANK copula
        elif self.cop_type == 'FRANK':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.cop_type)
            uu, vv = sample_frank(self.obs, self.theta, self.seed)

        # GUMBEL copula
        elif self.cop_type == 'GUMBEL':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.cop_type)
            uu, vv = sample_gumbel(self.obs, self.theta, self.seed)

        # GAUSSIAN copula
        elif self.cop_type == 'GAUSSIAN':
            assert hasattr(self, 'tau'), 'Please specify tau for %r copula' % (self.cop_type)

            xx = sample_gaussian(self.obs, self.tau, self.seed)

        # T-Copula
        elif self.cop_type == 'TDISTR':
            assert hasattr(self, 'tau'), 'Please specify tau for %r copula' % (self.cop_type)
            assert hasattr(self, 'df'), 'Please specify df for %r copula' % (self.cop_type)
            xx = sample_tdistr(self.obs, self.tau, self.df, self.seed)

        if self.cop_type not in ['GAUSSIAN', 'TDISTR']:
            xx = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

        assert xx.all() >= 0 & xx.all() <= 1

        # Apply inverse Sigmoid
        xx = special.logit(xx)

        self.xx = xx


def split_train_val_test(xx, seed):
    train, testval = model_selection.train_test_split(xx, random_state=seed, test_size=0.2)
    val, test = model_selection.train_test_split(testval, random_state=seed, test_size=0.5)
    return train, val, test


def sample_clayton(obs, theta, seed, uu=None, ww=None):
    np.random.seed(seed)

    if uu is None:
        uu = np.random.uniform(size=obs)
        ww = np.random.uniform(size=obs)

    if theta <= -1:
        raise ValueError('the parameter for clayton copula should be more than -1')
    elif theta == 0:
        raise ValueError('The parameter for clayton copula should not be 0')

    if theta < sys.float_info.epsilon:
        vv = ww
    else:
        vv = uu * (ww**(-theta / (1 + theta)) - 1 + uu**theta)**(-1 / theta)
    return uu, vv


def sample_frank(obs, theta, seed, uu=None, ww=None):
    np.random.seed(seed)

    if uu is None:
        uu = np.random.uniform(size=obs)
        ww = np.random.uniform(size=obs)

    if theta == 0:
        raise ValueError('The parameter for frank copula should not be 0')

    if abs(theta) > np.log(sys.float_info.max):
        vv = (uu < 0) + np.sign(theta) * uu
    elif abs(theta) > np.sqrt(sys.float_info.epsilon):
        vv = -np.log((
            np.exp(-theta * uu) * (1 - ww) / ww + np.exp(
                -theta)) / (1 + np.exp(-theta * uu) * (1 - ww) / ww)) / theta
    else:
        vv = ww
    return uu, vv


def sample_gumbel(obs, theta, seed, uu=None, ww=None):
    np.random.seed(seed)

    if theta <= 1:
        raise ValueError('the parameter for GUMBEL copula should be greater than 1')
    if theta < 1 + sys.float_info.epsilon:
        if uu is None:
            uu = np.random.uniform(size=obs)
            ww = np.random.uniform(size=obs)
    else:
        u_int = np.random.uniform(size=obs)
        ww = np.random.uniform(size=obs)
        w1 = np.random.uniform(size=obs)
        w2 = np.random.uniform(size=obs)

        u_int = (u_int - 0.5) * np.pi
        u2 = u_int + np.pi / 2
        ee = -np.log(ww)
        tt = np.cos(u_int - u2 / theta) / ee
        gamma = (np.sin(u2 / theta) / tt)**(1 / theta) * tt / np.cos(u_int)
        s1 = (-np.log(w1))**(1 / theta) / gamma
        s2 = (-np.log(w2))**(1 / theta) / gamma
        uu = np.array(np.exp(-s1))
        vv = np.array(np.exp(-s2))
    return uu, vv


def sample_gaussian(obs, tau, seed):
    np.random.seed(seed)

    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., tau],
                                                         [tau, 1.]])
    xx = mvnorm.rvs(obs)
    norm = stats.norm()
    x_unif = norm.cdf(xx)
    return x_unif


def sample_tdistr(obs, tau, df, seed):
    np.random.seed(seed)

    x = multivariate_t(mu=[0, 0], sigma=[[1., tau],
                                         [tau, 0.5]], dof=df, m=obs)

    tt = stats.t(df=2)
    x_unif = tt.cdf(x)
    return x_unif


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
    z = np.random.multivariate_normal(np.zeros(d), sigma, m)
    return mu + z / np.sqrt(g)
