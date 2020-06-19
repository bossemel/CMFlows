import numpy as np
import sys
import scipy
from utils.split_train_test import split_train_val_test


class Copula_sampler:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args, transform=True):

        self.copula = args.copula
        if args.tau:
            self.tau = args.tau
        if args.theta:
            self.theta = args.theta
        if args.df:
            self.df = args.df
        self.obs = args.obs
        self.transform_fct = args.transform_fct

        Copula_sampler.sample_copulas(self, transform)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def sample_copulas(self, transform):
        """Produce obs samples of 2-dimensional Copula density distribution
        """
        assert self.copula in ['clayton', 'frank', 'gumbel', 'gaussian', 'tdistr'], \
            "%r is not a valid copula, choose from %r" % (self.copula, ['clayton', 'frank', 'gumbel', 'gaussian', 'tdistr'])

        # Following Copula definitions from
        # https://pydoc.net/copulalib/1.1.0/copulalib.copulalib/
        # Conditional Distribution Method:
        # clayton copula
        if self.copula == 'clayton':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.copula)
            uu, vv = sample_clayton(self.obs, self.theta)

        # frank copula
        elif self.copula == 'frank':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.copula)
            uu, vv = sample_frank(self.obs, self.theta)

        # gumbel copula
        elif self.copula == 'gumbel':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.copula)
            uu, vv = sample_gumbel(self.obs, self.theta)

        # gaussian copula
        elif self.copula == 'gaussian':
            assert hasattr(self, 'tau'), 'Please specify tau for %r copula' % (self.copula)

            xx = sample_gaussian(self.obs, self.tau)

        # T-Copula
        elif self.copula == 'tdistr':
            assert hasattr(self, 'tau'), 'Please specify tau for %r copula' % (self.copula)
            assert hasattr(self, 'df'), 'Please specify df for %r copula' % (self.copula)
            xx = sample_tdistr(self.obs, self.tau, self.df)

        if self.copula not in ['gaussian', 'tdistr']:
            xx = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

        assert xx.all() > 0 & xx.all() < 1

        # Apply inverse Sigmoid
        if transform:
            if self.transform_fct == 'sigmoid':
                xx = scipy.special.logit(xx)
            if self.transform_fct == 'gaussian':
                norm = scipy.stats.norm()
                xx = norm.ppf(xx)

        self.xx = xx


def sample_clayton(obs, theta, uu=None, ww=None):
    """Sample from clayton copula density

    Params:
        obs: how many samples to generate
        theta: clayton copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
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


def sample_frank(obs, theta, uu=None, ww=None):
    """Sample from frank copula density

    Params:
        obs: how many samples to generate
        theta: frank copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
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


def sample_gumbel(obs, theta, uu=None, ww=None):
    """Sample from gumbel copula density

    Params:
        obs: how many samples to generate
        theta: gumbel copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if theta <= 1:
        raise ValueError('the parameter for gumbel copula should be greater than 1')
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


def sample_gaussian(obs, tau):
    """Sample from gaussian copula density

    Params:
        obs: how many samples to generate
        tau: gaussian copula parameter

    Returns:
        x_unif: samples
    """
    mvnorm = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., tau],
                                                               [tau, 1.]])
    xx = mvnorm.rvs(obs)
    norm = scipy.stats.norm()
    x_unif = norm.cdf(xx)
    return x_unif


def sample_tdistr(obs, tau, df):
    """Sample from t-distr copula density

    Params:
        obs: how many samples to generate
        theta: t-distr copula parameter
        uu, ww: fixed input grid

    Returns:
        x_unif: samples
    """
    x = multivariate_t(mu=[0, 0], sigma=[[1., tau],
                                         [tau, 0.5]], dof=df, m=obs)

    tt = scipy.stats.t(df=2)
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

    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    d = len(sigma)
    g = np.tile(np.random.gamma(dof / 2, 2 / dof, m), (d, 1)).T
    z = np.random.multivariate_normal(np.zeros(d), sigma, m)
    return mu + z / np.sqrt(g)


def _g(theta, z):
    r"""Helper function to solve frank copula.
    This functions encapsulates :math:`g(z) = e^{-\theta z} - 1` used on frank copulas.
    Argument:
        z: np.ndarray
    Returns:
        np.ndarray
    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    return np.exp(np.multiply(-theta, z)) - 1


def gumbel_cdf(theta, uu, vv):
    r"""Compute the cumulative distribution function for the gumbel copula.
    The cumulative density(cdf), or distribution function for the gumbel family of copulas
    correspond to the formula:
    .. math:: C(u,v) = e^{-((-\ln u)^{\theta} + (-\ln v)^{\theta})^{\frac{1}{\theta}}}
    Args:
        X (np.ndarray)
    Returns:
        np.ndarray: cumulative probability for the given datapoints, cdf(X).
    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    if theta == 1:
        return np.multiply(uu, vv)

    else:
        h = np.power(-np.log(uu), theta) + np.power(-np.log(vv), theta)
        h = -np.power(h, 1.0 / theta)
        cdfs = np.exp(h)
        return cdfs


def copula_pdf(copula, theta, uu, vv):
    """Compute probability density function for given copula family.
    Args:
        X (numpy.ndarray)
    Returns:
        numpy.ndarray: Probability density for the input values.
    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    if copula == 'clayton':
        a = (theta + 1) * np.power(np.multiply(uu, vv), -(theta + 1))
        b = np.power(uu, -theta) + np.power(vv, -theta) - 1
        c = -(2 * theta + 1) / theta
        pdf = a * np.power(b, c)
        assert pdf.all() > 0 & pdf.all() < 1
        return pdf
    if copula == 'frank':
        if theta == 0:
            return np.multiply(uu, vv)

        else:
            num = np.multiply(np.multiply(-theta, _g(theta, 1)), 1 + _g(theta, np.add(uu, vv)))
            aux = np.multiply(_g(theta, uu), _g(theta, vv)) + _g(theta, 1)
            den = np.power(aux, 2)
            pdf = num / den
            assert pdf.all() > 0 & pdf.all() < 1
            return pdf
    if copula == 'gumbel':
        if theta == 1:
            return np.multiply(uu, vv)

        else:
            a = np.power(np.multiply(uu, vv), -1)
            tmp = np.power(-np.log(uu), theta) + np.power(-np.log(vv), theta)
            b = np.power(tmp, -2 + 2.0 / theta)
            c = np.power(np.multiply(np.log(uu), np.log(vv)), theta - 1)
            d = 1 + (theta - 1) * np.power(tmp, -1.0 / theta)
            pdf = gumbel_cdf(theta, uu, vv) * a * b * c * d
            assert pdf.all() >= 0 & pdf.all() <= 1
            return pdf
