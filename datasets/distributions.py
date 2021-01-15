import datasets.distributions
from utils import split_train_val_test, normalize
import numpy as np
import scipy.stats
import sys
import scipy
eps = 0.0001


def marginal_transform(inputs, marginal, mu=None, var=None, alpha=None):
    """Transforms the uniform copula marginals into a different distribution.

    Params:
        inputs: copula samples vector
        marginal: desired marginal distributions
        args: passed input arguments

    Returns:
        inputs: transformed samples vector
    """
    if marginal == 'gaussian':
        norm = scipy.stats.norm(loc=mu, scale=var)
        inputs = norm.ppf(inputs)
    elif marginal == 'uniform':
        return inputs
    elif marginal == 'lognormal':
        lognorm = scipy.stats.lognorm(s=0.5, loc=mu, scale=var)
        inputs = lognorm.ppf(inputs)
    elif marginal == 'gamma':
        gamma = scipy.stats.gamma(alpha)
        inputs = gamma.ppf(inputs)
    else:
        raise NotImplementedError
    return inputs


class Joint_Distr():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, copula, marginal_1, marginal_2, theta, obs, mu=None, var=None, alpha=None, no_val=False,
                 random_seed=None):

        self.mu = mu
        self.var = var
        self.alpha = alpha
        self.theta = theta

        self.copula = copula
        self.marginal_1 = marginal_1
        self.marginal_2 = marginal_2
        self.obs = obs

        Joint_Distr.sampler(self, obs=obs, random_seed=random_seed)

        if no_val:
            trn, tst = split_train_val_test(self.xx, only_val=no_val)
            self.trn = trn.astype(np.float32)
            self.tst = tst.astype(np.float32)
        else:
            trn, val, tst = split_train_val_test(self.xx)
            self.trn = trn.astype(np.float32)
            self.val = val.astype(np.float32)
            self.tst = tst.astype(np.float32)

    def sampler(self, obs=None, random_seed=None):
        """Returns copula samples.
        """
        copula_distr = datasets.distributions.Copula_Distr(self.copula,
                                                           self.theta,
                                                           obs=obs,
                                                           transform=False,
                                                           random_seed=random_seed)
        copula_distr.sampler(transform=False, obs=self.obs)
        assert not np.isnan(np.sum(copula_distr.xx))
        marginal_1 = marginal_transform(inputs=copula_distr.xx[:, 0:1],
                                        marginal=self.marginal_1,
                                        mu=self.mu,
                                        var=self.var,
                                        alpha=self.alpha)
        marginal_2 = marginal_transform(inputs=copula_distr.xx[:, 1:2],
                                        marginal=self.marginal_2,
                                        mu=self.mu,
                                        var=self.var,
                                        alpha=self.alpha)

        xx = np.concatenate([marginal_1, marginal_2], axis=1)
        assert not np.isnan(np.sum(xx))
        assert not np.isnan(np.sum(normalize(xx)))

        self.xx = normalize(xx)


class Marginals():
    """Class for univariate samples
    """
    def __init__(self, marginal, obs, mu=None, var=None, alpha=None, low=None, high=None, random_seed=None):

        self.marginal = marginal
        self.obs = obs
        self.mu = mu
        self.var = var
        self.alpha = alpha
        self.low = low
        self.high = high

        self.xx = Marginals.sampler(self, random_seed=random_seed)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = trn.astype(np.float32)
        self.val = val.astype(np.float32)
        self.tst = tst.astype(np.float32)

    def sampler(self, obs=None, random_seed=None):
        """Returns marginal samples.
        """
        if self.marginal == 'gaussian':
            assert self.mu is not None, 'Please specify mean mu for %r distribution' % (self.marginal)
            assert self.var is not None, 'Please specify variance var for %r distribution' % (self.marginal)
            dataset = scipy.stats.norm.rvs(loc=self.mu,
                                           scale=self.var,
                                           size=[self.obs if obs is None else obs],
                                           random_state=random_seed)
        elif self.marginal == 'uniform':
            assert hasattr(self, 'low'), 'Please specify lower bound a for %r distribution' % (self.marginal)
            assert hasattr(self, 'high'), 'Please specify upper bound b for %r distribution' % (self.marginal)

            dataset = scipy.stats.uniform.rvs(loc=self.low,
                                              scale=self.high,
                                              size=[self.obs if obs is None else obs],
                                              random_state=random_seed)
        elif self.marginal == 'gamma':
            assert self.alpha is not None, 'Please specify alpha for %r distribution' % (self.marginal)

            dataset = scipy.stats.gamma.rvs(a=self.alpha,
                                            size=[self.obs if obs is None else obs],
                                            random_state=random_seed)

        elif self.marginal == 'lognormal':
            assert hasattr(self, 'mu'), 'Please specify mu for %r distribution' % (self.marginal)
            assert hasattr(self, 'var'), 'Please specify var %r distribution' % (self.marginal)

            dataset = scipy.stats.lognorm.rvs(s=0.5,
                                              loc=self.mu,
                                              scale=self.var,
                                              size=[self.obs if obs is None else obs],
                                              random_state=random_seed)

        return normalize(dataset.reshape(-1, 1))

    def pdf(self, inputs):
        if self.marginal == 'gaussian':
            pdf_samples = scipy.stats.norm.pdf(inputs,
                                               loc=self.mu,
                                               scale=self.var)
        elif self.marginal == 'uniform':
            assert hasattr(self, 'low'), 'Please specify lower bound a for %r distribution' % (self.marginal)
            assert hasattr(self, 'high'), 'Please specify upper bound b for %r distribution' % (self.marginal)

            pdf_samples = scipy.stats.uniform.pdf(inputs,
                                                  loc=self.low,
                                                  scale=self.high)
        elif self.marginal == 'gamma':
            assert self.alpha is not None, 'Please specify %r for %r distribution' % (self.marginal)

            pdf_samples = scipy.stats.gamma.pdf(inputs,
                                                a=self.alpha)

        elif self.marginal == 'lognormal':
            pdf_samples = scipy.stats.lognorm.pdf(inputs,
                                                  s=0.5,
                                                  loc=self.mu,
                                                  scale=self.var)

        return pdf_samples


class Copula_Distr:
    def __init__(self, copula, theta, obs=None, transform=True, random_seed=None):

        self.copula = copula
        self.theta = theta
        self.obs = obs
        self.transform = transform

        Copula_Distr.sampler(self, self.transform, self.obs, random_seed=random_seed)
        trn, val, tst = split_train_val_test(self.xx)

        self.trn = trn.astype(np.float32)
        self.val = val.astype(np.float32)
        self.tst = tst.astype(np.float32)

    def sampler(self, transform=None, obs=None, random_seed=None):
        """Produce obs samples of 2-dimensional Copula density distribution
        """

        # Following Copula definitions from
        # https://pydoc.net/copulalib/1.1.0/copulalib.copulalib/
        # Conditional Distribution Method:
        # clayton copula
        if self.copula == 'clayton':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.copula)
            uu, vv = sample_clayton([self.obs if obs is None else obs], self.theta, random_seed=None)

        # frank copula
        elif self.copula == 'frank':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.copula)
            uu, vv = sample_frank([self.obs if obs is None else obs], self.theta, random_seed=None)

        # gumbel copula
        elif self.copula == 'gumbel':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % (self.copula)
            uu, vv = sample_gumbel([self.obs if obs is None else obs], self.theta, random_seed=None)

        xx = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

        assert xx.all() > 0 & xx.all() < 1

        # Apply inverse Gaussian
        if transform:
            norm = scipy.stats.norm()
            xx = norm.ppf(xx)

        self.xx = xx

    def pdf(self, xx):
        uu = xx[:, 0]
        vv = xx[:, 1]
        copula_pdf_samples = copula_pdf(self.copula, self.theta, uu, vv)
        return copula_pdf_samples


def sample_clayton(obs, theta, uu=None, ww=None, random_seed=None):
    """Sample from clayton copula density

    Params:
        obs: how many samples to generate
        theta: clayton copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if uu is None:
        if random_seed is None:
            uu = np.random.uniform(size=obs)
            ww = np.random.uniform(size=obs)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs)


    if theta <= -1:
        raise ValueError('the parameter for clayton copula should be more than -1')
    elif theta == 0:
        raise ValueError('The parameter for clayton copula should not be 0')

    if theta < sys.float_info.epsilon:
        vv = ww
    else:
        vv = uu * (ww**(-theta / (1 + theta)) - 1 + uu**theta)**(-1 / theta)
    return uu, vv


def sample_frank(obs, theta, uu=None, ww=None, random_seed=None):
    """Sample from frank copula density

    Params:
        obs: how many samples to generate
        theta: frank copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if uu is None:
        if random_seed is None:
            uu = np.random.uniform(size=obs)
            ww = np.random.uniform(size=obs)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs)

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


def sample_gumbel(obs, theta, uu=None, ww=None, random_seed=None):
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
        if random_seed is None:
            uu = np.random.uniform(size=obs)
            ww = np.random.uniform(size=obs)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs)
    else:
        if random_seed is None:
            u_int = np.random.uniform(size=obs)
            ww = np.random.uniform(size=obs)
            w1 = np.random.uniform(size=obs)
            w2 = np.random.uniform(size=obs)
        else:
            u_int = np.random.RandomState(random_seed).uniform(size=obs)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs)
            w1 = np.random.RandomState(random_seed + 2).uniform(size=obs)
            w2 = np.random.RandomState(random_seed + 3).uniform(size=obs)

        u_int = (u_int - 0.5) * np.pi
        u2 = u_int + np.pi / 2
        ee = -np.log(ww)
        tt = np.cos(u_int - u2 / theta) / ee
        gamma = (np.sin(u2 / theta) / tt)**(1 / theta) * tt / np.cos(u_int)
        s1 = (-np.log(w1))**(1 / theta) / gamma
        s2 = (-np.log(w2))**(1 / theta) / gamma
        uu = np.array(np.exp(-s1))
        vv = np.array(np.exp(-s2))
    assert not np.isnan(np.sum(uu))
    assert not np.isnan(np.sum(vv))
    assert uu.all() >= 0
    assert vv.all() >= 0
    return uu, vv


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


def remove_0_1(array):
    array[array == 0] = eps
    array[array == 1] = 1 - eps
    return array


def copula_pdf(copula, theta, uu, vv):
    """Compute probability density function for given copula family.
    Args:
        X (numpy.ndarray)
    Returns:
        numpy.ndarray: Probability density for the input values.
    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    uu = remove_0_1(uu)
    vv = remove_0_1(vv)
    assert np.min(uu) > 0 and np.max(uu) < 1, 'min: {}, max: {}'.format(np.min(uu), np.max(uu))
    assert np.min(vv) > 0 and np.max(vv) < 1, 'min: {}, max: {}'.format(np.min(vv), np.max(vv))

    if copula == 'clayton':
        assert not np.isnan(np.multiply(uu, vv).sum()), '{}'.format(np.multiply(uu, vv).sum())
        a = (theta + 1) * np.power(np.multiply(uu, vv), -(theta + 1))
        assert not np.isnan(a.sum())
        b = np.power(uu, -theta) + np.power(vv, -theta) - 1
        assert not np.isnan(b.sum())
        c = -(2 * theta + 1) / theta
        assert not np.isnan(c)
        assert not np.isnan(np.power(b, c).sum())
        pdf = a * np.power(b, c, dtype=np.float)
        assert np.min(pdf) > 0, 'clayton_{}_{}_b:{} c: {}'.format(np.min(pdf), theta, b, c)
        return pdf
    if copula == 'frank':
        if theta == 0:
            return np.multiply(uu, vv)

        else:
            num = np.multiply(np.multiply(-theta, _g(theta, 1)), 1 + _g(theta, np.add(uu, vv)))
            aux = np.multiply(_g(theta, uu), _g(theta, vv)) + _g(theta, 1)
            den = np.power(aux, 2, dtype=np.float)
            pdf = num / den
            assert np.min(pdf) >= 0, 'frank_{}'.format(np.min(pdf))
            return pdf
    if copula == 'gumbel':
        if theta == 1:
            return np.multiply(uu, vv)

        else:
            a = np.power(np.multiply(uu, vv), -1, dtype=np.float)
            assert not np.isnan(a.sum())
            tmp = np.power(-np.log(uu), theta) + np.power(-np.log(vv), theta)
            b = np.power(tmp, -2 + 2.0 / theta, dtype=np.float)
            assert not np.isnan(b.sum())

            c = np.power(np.multiply(np.log(uu), np.log(vv)), theta - 1)
            assert not np.isnan(c.sum())

            d = 1 + (theta - 1) * np.power(tmp, -1.0 / theta, dtype=np.float)
            pdf = gumbel_cdf(theta, uu, vv) * a * b * c * d
            assert np.min(pdf) >= 0, 'gumbel_{}'.format(np.min(pdf))
            return pdf
