from utils import split_train_val_test, normalize
import numpy as np
import scipy.stats
import sys
import scipy
import pynverse
import torch
import os
import pyvinecopulib as pv
from pathlib import Path

eps = 1e-07


def marginal_transform(inputs, marginal, mu_=None, var_=None, alpha_=None):
    """Transforms the uniform copula marginals into a different distribution.

    Params:
        inputs: copula samples vector
        marginal: desired marginal distributions
        args: passed input arguments

    Returns:
        inputs: transformed samples vector
    """
    if marginal == 'gaussian':
        norm = scipy.stats.norm(loc=mu_, scale=var_)
        inputs = norm.ppf(inputs)
    elif marginal == 'uniform':
        return inputs
    elif marginal == 'lognormal':
        lognorm = scipy.stats.lognorm(s=0.5, loc=mu_, scale=var_)
        inputs = lognorm.ppf(inputs)
    elif marginal == 'gamma':
        gamma = scipy.stats.gamma(alpha_)
        inputs = gamma.ppf(inputs)
    elif marginal == 'gmm':
        distr_1 = lambda xx: scipy.stats.norm.cdf(xx, loc=mu_ - 2, scale=var_ * 2)
        distr_2 = lambda xx: scipy.stats.norm.cdf(xx, loc=mu_ + 2, scale=var_ / 2)
        distr_3 = lambda xx: scipy.stats.norm.cdf(xx, loc=mu_, scale=var_ / 4)
    elif marginal == 'mix_gamma':
        distr_1 = lambda xx: scipy.stats.gamma.cdf(xx, 1)
        distr_2 = lambda xx: scipy.stats.gamma.cdf(xx, 5)
        distr_3 = lambda xx: scipy.stats.gamma.cdf(xx, 2)
    elif marginal == 'mix_lognormal':
        distr_1 = lambda xx: scipy.stats.lognorm.cdf(xx, s=0.1, loc=mu_ - 2, scale=var_ * 2)
        distr_2 = lambda xx: scipy.stats.lognorm.cdf(xx, s=0.9, loc=mu_ + 2, scale=var_ / 2)
        distr_3 = lambda xx: scipy.stats.lognorm.cdf(xx, s=0.5, loc=mu_, scale=var_)
    elif marginal == 'mix_gauss_gamma':
        distr_1 = lambda xx: scipy.stats.norm.cdf(xx, loc=mu_, scale=var_ / 5)
        distr_2 = lambda xx: scipy.stats.gamma.cdf(xx, alpha_)
        distr_3 = lambda xx: scipy.stats.gamma.cdf(xx, alpha_ * 5)
    if marginal in ['gmm', 'mix_gamma', 'mix_lognormal', 'mix_gauss_gamma']:
        inverse_cdf = pynverse.inversefunc(lambda xx: 0.4 * distr_1(xx) + 0.4 * distr_2(xx) + 0.2 * distr_3(xx))
        inputs = inverse_cdf(inputs)
    return inputs


class Joint_Distr:
    """Class for bivariate samples given a copula correlation and individual marginals.
    """
    def __init__(self, copula_, marginal_1_, marginal_2_, theta_, obs_, mu_=None, var_=None, alpha_=None, no_val=False,
                 random_seed=None):

        self.mu = mu_
        self.var = var_
        self.alpha = alpha_
        self.theta = theta_

        self.copula = copula_
        self.marginal_1 = marginal_1_
        self.marginal_2 = marginal_2_
        self.obs = obs_

        Joint_Distr.sampler(self, obs_=obs_, random_seed=random_seed)

        if no_val:
            trn, tst = split_train_val_test(self.xx, only_val=no_val)
            self.trn = trn.astype(np.float32)
            self.tst = tst.astype(np.float32)
        else:
            trn, val, tst = split_train_val_test(self.xx)
            self.trn = trn.astype(np.float32)
            self.val = val.astype(np.float32)
            self.tst = tst.astype(np.float32)

    def sampler(self, obs_=None, random_seed=None):
        """Returns copula samples.
        """
        copula_distr = Copula_Distr(self.copula,
                                    self.theta,
                                    obs_)
        copula_distr.sampler(obs_=obs_, transform=False)

        xx = marginal_transform(inputs=copula_distr.xx,
                                marginal=self.marginal_1,
                                mu_=self.mu,
                                var_=self.var,
                                alpha_=self.alpha)

        self.xx = normalize(xx)


class Marginals:
    """Class for univariate samples
    """
    def __init__(self, marginal, obs_, mu_=None, var_=None, alpha_=None, low_=None, high_=None, random_seed=None):

        self.marginal = marginal
        self.obs = obs_
        self.mu = mu_
        self.var = var_
        self.alpha = alpha_
        self.low = low_
        self.high = high_

        self.xx = Marginals.sampler(self)

        trn, val, tst = split_train_val_test(self.xx)

        self.trn = trn.astype(np.float32)
        self.val = val.astype(np.float32)
        self.tst = tst.astype(np.float32)

    def sampler(self):
        """Returns marginal samples.
        """
        dataset = scipy.stats.uniform.rvs(size=self.obs)
        dataset = marginal_transform(dataset, self.marginal, mu_=self.mu, var_=self.var, alpha_=self.alpha)
        return normalize(dataset.reshape(-1, 1))

    def pdf(self, inputs):
        if self.marginal == 'gaussian':
            pdf_samples = scipy.stats.norm.pdf(inputs,
                                               loc=self.mu,
                                               scale=self.var)
        elif self.marginal == 'uniform':
            assert hasattr(self, 'low'), 'Please specify lower bound a for %r distribution' % self.marginal
            assert hasattr(self, 'high'), 'Please specify upper bound b for %r distribution' % self.marginal

            pdf_samples = scipy.stats.uniform.pdf(inputs,
                                                  loc=self.low,
                                                  scale=self.high)
        elif self.marginal == 'gamma':
            assert self.alpha is not None, 'Please specify %r for %r distribution' % self.marginal

            pdf_samples = scipy.stats.gamma.pdf(inputs,
                                                a=self.alpha)

        elif self.marginal == 'lognormal':
            pdf_samples = scipy.stats.lognorm.pdf(inputs,
                                                  s=0.5,
                                                  loc=self.mu,
                                                  scale=self.var)

        return pdf_samples


class Copula_Distr:
    def __init__(self, copula_, theta_, obs_, transform_=True):

        self.copula = copula_
        self.theta = theta_
        self.obs = obs_
        self.transform = transform_

        Copula_Distr.sampler(self, self.transform, self.obs)
        trn, val, tst = split_train_val_test(self.xx)

        self.trn = trn.astype(np.float32)
        self.val = val.astype(np.float32)
        self.tst = tst.astype(np.float32)

    def sampler(self, transform=None, obs_=None):
        """Produce obs_ samples of 2-dimensional Copula density distribution
        """
        # Following Copula definitions from
        # https://pydoc.net/copulalib/1.1.0/copulalib.copulalib/
        # Conditional Distribution Method:
        # clayton copula
        if self.copula == 'clayton':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % self.copula
            uu, vv = sample_clayton([self.obs if obs_ is None else obs_], self.theta, random_seed=None)

        # frank copula
        elif self.copula == 'frank':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % self.copula
            uu, vv = sample_frank([self.obs if obs_ is None else obs_], self.theta, random_seed=None)

        # gumbel copula
        elif self.copula == 'gumbel':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % self.copula
            uu, vv = sample_gumbel([self.obs if obs is None else obs], self.theta, random_seed=None)

        # gumbel copula
        elif self.copula == 'independent':
            xx = scipy.stats.uniform.rvs(size=(self.obs, 2))

        if self.copula in ['clayton', 'frank', 'gumbel']:
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


def sample_clayton(obs_, theta, uu=None, ww=None, random_seed=None):
    """Sample from clayton copula density

    Params:
        obs_: how many samples to generate
        theta: clayton copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if uu is None:
        if random_seed is None:
            uu = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)


    if theta <= -1:
        raise ValueError('the parameter for clayton copula should be more than -1')
    elif theta == 0:
        raise ValueError('The parameter for clayton copula should not be 0')

    if theta < sys.float_info.epsilon:
        vv = ww
    else:
        vv = uu * (ww**(-theta / (1 + theta)) - 1 + uu**theta)**(-1 / theta)
    return uu, vv


def sample_frank(obs_, theta, uu=None, ww=None, random_seed=None):
    """Sample from frank copula density

    Params:
        obs_: how many samples to generate
        theta: frank copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if uu is None:
        if random_seed is None:
            uu = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)

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


def sample_gumbel(obs_, theta_, random_seed=None):
    """Sample from gumbel copula density

    Params:
        obs_: how many samples to generate
        theta_: gumbel copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if theta_ <= 1:
        raise ValueError('the parameter for gumbel copula should be greater than 1')
    if theta_ < 1 + sys.float_info.epsilon:
        if random_seed is None:
            uu = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)
    else:
        if random_seed is None:
            u_int = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
            w1 = np.random.uniform(size=obs_)
            w2 = np.random.uniform(size=obs_)
        else:
            u_int = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)
            w1 = np.random.RandomState(random_seed + 2).uniform(size=obs_)
            w2 = np.random.RandomState(random_seed + 3).uniform(size=obs_)

        u_int = (u_int - 0.5) * np.pi
        u2 = u_int + np.pi / 2
        ee = -np.log(ww)
        tt = np.cos(u_int - u2 / theta_) / ee
        gamma = (np.sin(u2 / theta_) / tt)**(1 / theta_) * tt / np.cos(u_int)
        s1 = (-np.log(w1))**(1 / theta_) / gamma
        s2 = (-np.log(w2))**(1 / theta_) / gamma
        uu = np.array(np.exp(-s1))
        vv = np.array(np.exp(-s2))
    assert not np.isnan(np.sum(uu))
    assert not np.isnan(np.sum(vv))
    assert uu.all() >= 0
    assert vv.all() >= 0
    return uu, vv


def _g(theta_, z):
    r"""Helper function to solve frank copula.
    This functions encapsulates :math:`g(z) = e^{-\theta_ z} - 1` used on frank copulas.
    Argument:
        z: np.ndarray
    Returns:
        np.ndarray
    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    return np.exp(np.multiply(-theta_, z)) - 1


def gumbel_cdf(theta_, uu, vv):
    if theta_ == 1:
        return np.multiply(uu, vv)

    else:
        h = np.power(-np.log(uu), theta_) + np.power(-np.log(vv), theta_)
        h = -np.power(h, 1.0 / theta_)
        cdfs = np.exp(h)
        return cdfs


def remove_0_1(array):
    array[array == 0] = eps
    array[array == 1] = 1 - eps
    return array


def copula_pdf(copula_, theta_, uu, vv):
    uu = remove_0_1(uu).astype('float64')
    vv = remove_0_1(vv).astype('float64')
    assert np.min(uu) > 0 and np.max(uu) < 1, 'min: {}, max: {}'.format(np.min(uu), np.max(uu))
    assert np.min(vv) > 0 and np.max(vv) < 1, 'min: {}, max: {}'.format(np.min(vv), np.max(vv))

    if copula_ == 'clayton':
        a = (theta_ + 1) * np.power(np.multiply(uu, vv), -(theta_ + 1))
        assert np.isfinite(a.sum()), 'np.multiply(uu, vv): {}, -(theta_ + 1): {}'.format(np.multiply(uu, vv).dtype, type(-(theta_ + 1)))
        b = np.power(uu, -theta_) + np.power(vv, -theta_) - 1
        c = -(2 * theta_ + 1) / theta_
        pdf = a * np.power(b, c, dtype=np.float)
        assert np.min(pdf) > 0, 'clayton_{}_{}_b:{} c: {}'.format(np.min(pdf), theta_, b, c)
        return pdf
    if copula_ == 'frank':
        if theta_ == 0:
            return np.multiply(uu, vv)

        else:
            num = np.multiply(np.multiply(-theta_, _g(theta_, 1)), 1 + _g(theta_, np.add(uu, vv)))
            aux = np.multiply(_g(theta_, uu), _g(theta_, vv)) + _g(theta_, 1)
            den = np.power(aux, 2, dtype=np.float)
            pdf = num / den
            assert np.min(pdf) >= 0, 'frank_{}'.format(np.min(pdf))
            return pdf
    if copula_ == 'gumbel':
        if theta_ == 1:
            return np.multiply(uu, vv)

        else:
            a = np.power(np.multiply(uu, vv), -1, dtype=np.float)
            tmp = np.power(-np.log(uu), theta_) + np.power(-np.log(vv), theta_)
            b = np.power(tmp, -2 + 2.0 / theta_, dtype=np.float)

            c = np.power(np.multiply(np.log(uu), np.log(vv)), theta_ - 1)

            d = 1 + (theta_ - 1) * np.power(tmp, -1.0 / theta_, dtype=np.float)
            pdf = gumbel_cdf(theta_, uu, vv) * a * b * c * d
            assert np.min(pdf) >= 0, 'gumbel_{}'.format(np.min(pdf))
            return pdf
    else:
        raise NotImplementedError


def gen_mv_copula(mix, copula='clayton', marginal='gamma', obs=10000, random_seed=4, disable_marginal=False):
    if mix is False:
        if copula == 'clayton':
            pair_copula = pv.BicopFamily.clayton
            theta = 2
        elif copula == 'frank':
            pair_copula = pv.BicopFamily.frank
            theta = 5
        elif copula == 'gumbel':
            pair_copula = pv.BicopFamily.gumbel
            theta = 5

        # Specify pair-copulas
        bicop = pv.Bicop(family=pair_copula, parameters=[theta])
        pcs = [[bicop, bicop, bicop], [bicop, bicop], [bicop]]
    else:
        bicop_1 = pv.Bicop(family=pv.BicopFamily.gumbel, parameters=[5])
        bicop_2 = pv.Bicop(family=pv.BicopFamily.clayton, parameters=[2])
        bicop_3 = pv.Bicop(family=pv.BicopFamily.frank, parameters=[5])
        pcs = [[bicop_1, bicop_2, bicop_3], [bicop_1, bicop_2], [bicop_1]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 0, 0], [4, 0, 0, 0]])

    # Set-up a vine copula
    copula = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    copula_samples = copula.simulate(n=obs, seeds=[random_seed])
    if not disable_marginal:
        for dim in range(copula_samples.shape[1]):
            copula_samples[:, dim] = normalize(marginal_transform(copula_samples[:, dim], marginal=marginal,
                                                                  mu_=mu, var_=var, alpha_=alpha))
    assert not np.isnan(np.sum(copula_samples)), '{}'.format(copula_samples[np.isnan(copula_samples)])
    return torch.from_numpy(copula_samples)


def save_dataset_2D(copula, marginal_1, marginal_2, theta, obs, mu, var, alpha, random_seed):
    dataset = Joint_Distr(copula, marginal_1, marginal_2, theta, obs,
                                                 mu_=mu, var_=var, alpha_=alpha, random_seed=random_seed)

    torch.save(dataset.trn, os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_trn'.format(copula, marginal_1, marginal_2)))
    torch.save(dataset.val, os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_val'.format(copula, marginal_1, marginal_2)))
    torch.save(dataset.tst, os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_tst'.format(copula, marginal_1, marginal_2)))


def save_dataset_4D(mix, copula='clayton', marginal='gamma', obs=10000, random_seed=4):
    dataset = gen_mv_copula(mix, copula, marginal, obs, random_seed, disable_marginal=False)

    torch.save(dataset, os.path.join(os.path.join('datasets', 'joint_data'), '4D_{}_{}_mix{}'.format(copula, marginal, mix)))


if __name__ == '__main__':
    path = os.path.join('datasets', 'joint_data')
    Path(path).mkdir(parents=True, exist_ok=True)
    copula_list = ['clayton', 'frank', 'gumbel', 'independent']
    marginal_1_list = ['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal', 'mix_gauss_gamma']
    marginal_2_list = ['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal', 'mix_gauss_gamma']
    alpha = 10
    mu = 0
    var = 1
    obs = 10000
    seed = 4

    for copula in copula_list:
        for marginal_1 in marginal_1_list:
            #for marginal_2 in marginal_2_list:
            if copula == 'clayton':
                theta = 2
            else:
                theta = 5
            print('Creating 2D dataset for {} copula with {} marginal..'.format(copula, marginal_1))
            save_dataset_2D(copula, marginal_1, marginal_1, theta, obs=obs, mu=mu, var=var, alpha=alpha, random_seed=seed)
            print('Creating 4D dataset for {} copula with {} marginal..'.format(copula, marginal_1))
            save_dataset_4D(False, copula, marginal_1, obs, random_seed=seed)

    for marginal_1 in marginal_1_list:
        print('Creating 4D dataset for {} copula with mixed marginals..'.format(copula))
        save_dataset_4D(True, marginal=marginal_1, obs=obs, random_seed=seed)

    save_dataset_2D('frank', 'mix_gamma', 'mix_gamma', 5, obs=obs, mu=mu, var=var, alpha=alpha, random_seed=seed)
