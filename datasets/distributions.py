import datasets.distributions
from utils.split_train_test import split_train_val_test
import numpy as np
import scipy.stats
import sys
import scipy


def marginal_transform(inputs, marginal, args):
    """Transforms the uniform copula marginals into a different distribution.

    Params:
        inputs: copula samples vector
        marginal: desired marginal distributions
        args: passed input arguments

    Returns:
        inputs: transformed samples vector
    """
    if marginal == 'gaussian':
        assert hasattr(args, 'mu') is not None, 'Please specify mean mu for %r distribution' % (args.marginal)
        assert hasattr(args, 'var') is not None, 'Please specify mean var for %r distribution' % (args.marginal)
        norm = scipy.stats.norm(loc=args.mu, scale=args.var)
        inputs = norm.ppf(inputs)
    elif marginal == 'uniform':
        return inputs
    elif marginal == 'lognormal':
        assert hasattr(args, 'mu') is not None, 'Please specify mean mu for %r distribution' % (args.marginal)
        assert hasattr(args, 'var') is not None, 'Please specify mean var for %r distribution' % (args.marginal)

        lognorm = scipy.stats.lognorm(s=0.5, loc=args.mu, scale=args.var)
        inputs = lognorm.ppf(inputs)
    elif marginal == 'gamma':
        assert hasattr(args, 'alpha') is not None, 'Please specify alpha for %r distribution' % (args.marginal)
        gamma = scipy.stats.gamma(args.alpha)
        inputs = gamma.ppf(inputs)
    elif marginal == 'bimodal_gaussian':
        inputs_split = np.split(inputs, 2)
        inputs_1 = inputs_split[0]
        inputs_2 = inputs_split[1]

        samples_1 = scipy.stats.norm.ppf(q=inputs_1, loc=2, scale=2)
        samples_2 = scipy.stats.norm.ppf(q=inputs_2, loc=12, scale=2)

        inputs = np.concatenate([samples_1, samples_2])
    else:
        raise NotImplementedError
    return inputs


class Joint_Distr():
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):
        self.xx = Joint_Distr.sampler(self, args)
        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def sampler(self, args, obs=None):
        """Returns copula samples.
        """
        copula_xx = datasets.distributions.Copula_Distr.sampler(args=args, transform=False)
        marginal_1 = marginal_transform(inputs=copula_xx[:, 0], marginal=args.marginal_1, args=args)
        marginal_2 = marginal_transform(inputs=copula_xx[:, 1], marginal=args.marginal_2, args=args)

        xx = np.concatenate([marginal_1.reshape(-1, 1), marginal_2.reshape(-1, 1)], axis=1)
        return xx


class Marginals():
    """Class for univariate samples
    """
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args):
        self.xx = Marginals.sampler(self, args)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = args.obs

    def sampler(self, args, obs=None):
        """Returns marginal samples.
        """
        if args.marginal == 'gaussian':
            assert args.mu is not None, 'Please specify mean mu for %r distribution' % (args.marginal)
            assert args.var is not None, 'Please specify variance var for %r distribution' % (args.marginal)
            dataset = scipy.stats.norm.rvs(loc=args.mu,
                                           scale=args.var,
                                           size=[args.obs if obs is None else obs])
        elif args.marginal == 'uniform':
            assert hasattr(args, 'low'), 'Please specify lower bound a for %r distribution' % (args.marginal)
            assert hasattr(args, 'high'), 'Please specify upper bound b for %r distribution' % (args.marginal)

            dataset = scipy.stats.uniform.rvs(loc=args.low,
                                              scale=args.high,
                                              size=[args.obs if obs is None else obs])
        elif args.marginal == 'gamma':
            assert args.alpha is not None, 'Please specify %r for %r distribution' % (args.marginal)

            dataset = scipy.stats.gamma.rvs(a=args.alpha, size=[args.obs if obs is None else obs])

        elif args.marginal == 'lognormal':
            assert hasattr(args, 'mu'), 'Please specify mu for %r distribution' % (args.marginal)
            assert hasattr(args, 'var'), 'Please specify var %r distribution' % (args.marginal)

            dataset = scipy.stats.lognorm.rvs(s=0.5,
                                              loc=args.mu,
                                              scale=args.var,
                                              size=[args.obs if obs is None else obs])

        elif args.marginal == 'bimodal_gaussian':
            samples_1 = scipy.stats.norm.rvs(loc=2, scale=2, size=[int(args.obs / 2) if obs is None else int(obs / 2)])
            samples_2 = scipy.stats.norm.rvs(loc=12, scale=2, size=[int(args.obs / 2) if obs is None else int(obs / 2)])

            dataset = np.concatenate([samples_1, samples_2])
        return dataset.reshape(-1, 1)

    def pdf(self, args, inputs):
        if args.marginal == 'gaussian':
            pdf_samples = scipy.stats.norm.pdf(inputs,
                                               loc=args.mu,
                                               scale=args.var)
        elif args.marginal == 'uniform':
            assert hasattr(args, 'low'), 'Please specify lower bound a for %r distribution' % (args.marginal)
            assert hasattr(args, 'high'), 'Please specify upper bound b for %r distribution' % (args.marginal)

            pdf_samples = scipy.stats.uniform.pdf(inputs,
                                                  loc=args.low,
                                                  scale=args.high)
        elif args.marginal == 'gamma':
            assert args.alpha is not None, 'Please specify %r for %r distribution' % (args.marginal)

            pdf_samples = scipy.stats.gamma.pdf(inputs,
                                                a=args.alpha)

        elif args.marginal == 'lognormal':
            pdf_samples = scipy.stats.lognorm.pdf(inputs,
                                                  s=0.5,
                                                  loc=args.mu,
                                                  scale=args.var)

        elif args.marginal == 'bimodal_gaussian':
            inputs_split = np.split(inputs, 2)
            inputs_1 = inputs_split[0]
            inputs_2 = inputs_split[1]

            samples_1 = scipy.stats.norm.pdf(inputs_1,
                                             loc=2,
                                             scale=2)
            samples_2 = scipy.stats.norm.pdf(inputs_2,
                                             loc=12,
                                             scale=2)

            pdf_samples = np.concatenate([samples_1, samples_2])
        return pdf_samples


class Copula_Distr:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, args, transform=True):

        self.xx = Copula_Distr.sampler(args, transform)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def sampler(args, transform, obs=None):
        """Produce obs samples of 2-dimensional Copula density distribution
        """

        # Following Copula definitions from
        # https://pydoc.net/copulalib/1.1.0/copulalib.copulalib/
        # Conditional Distribution Method:
        # clayton copula
        if args.copula == 'clayton':
            assert hasattr(args, 'theta'), 'Please specify theta for %r copula' % (args.copula)
            uu, vv = sample_clayton([args.obs if obs is None else obs], args.theta)

        # frank copula
        elif args.copula == 'frank':
            assert hasattr(args, 'theta'), 'Please specify theta for %r copula' % (args.copula)
            uu, vv = sample_frank([args.obs if obs is None else obs], args.theta)

        # gumbel copula
        elif args.copula == 'gumbel':
            assert hasattr(args, 'theta'), 'Please specify theta for %r copula' % (args.copula)
            uu, vv = sample_gumbel([args.obs if obs is None else obs], args.theta)

        # gaussian copula
        elif args.copula == 'gaussian':
            assert hasattr(args, 'tau'), 'Please specify tau for %r copula' % (args.copula)

            xx = sample_gaussian([args.obs if obs is None else obs], args.tau)

        # T-Copula
        elif args.copula == 'tdistr':
            assert hasattr(args, 'tau'), 'Please specify tau for %r copula' % (args.copula)
            assert hasattr(args, 'df'), 'Please specify df for %r copula' % (args.copula)
            xx = sample_tdistr([args.obs if obs is None else obs], args.tau, args.df)

        if args.copula not in ['gaussian', 'tdistr']:
            xx = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

        assert xx.all() > 0 & xx.all() < 1

        # Apply inverse Sigmoid
        if transform:
            if args.transform_fct == 'sigmoid':
                xx = scipy.special.logit(xx)
            if args.transform_fct == 'gaussian':
                norm = scipy.stats.norm()
                xx = norm.ppf(xx)

        return xx


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


class copula_distr():
    def __init__(self, copula, theta):
        self.copula = copula
        self.theta = theta

    def pdf(self, xx):
        uu = xx[:, 0]
        vv = xx[:, 1]
        return copula_pdf(self.copula, self.theta, uu, vv)


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
