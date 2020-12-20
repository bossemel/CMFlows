import datasets.distributions
import numpy as np
import random
from utils import js_divergence
import unittest


class Test_Copula_PDF(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_Copula_PDF, self).__init__(*args, **kwargs)
        self.obs = 10000
        self.theta = 2
        self.transform_fct = 'gaussian'
        self.copula = 'clayton'

    def test_one_copula_same_sample(self):
        copula_distr = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr.sampler(obs=self.obs)
        xx = copula_distr.xx
        X_in_p = copula_distr.pdf(xx)
        X_in_q = copula_distr.pdf(xx)
        jsd_X = js_divergence(X_in_p, X_in_q, X_in_p, X_in_q)
        self.assertEqual(jsd_X, 0)

    def test_different_theta(self):
        copula_distr_1 = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr_1.sampler(obs=self.obs)
        xx_1 = copula_distr_1.xx
        self.theta = 1.5
        copula_distr_2 = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr_2.sampler(obs=self.obs)
        xx_2 = copula_distr_2.xx

        X_in_p = copula_distr_1.pdf(xx_1)
        X_in_q = copula_distr_2.pdf(xx_1)
        Y_in_p = copula_distr_1.pdf(xx_2)
        Y_in_q = copula_distr_2.pdf(xx_2)
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertTrue(jsd_X_Y > 0 and jsd_X_Y <= 0.01)

    def test_clayton_frank(self):
        self.theta = 2
        copula_distr_1 = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr_1.sampler(obs=self.obs)
        xx_1 = copula_distr_1.xx
        self.copula = 'frank'
        copula_distr_2 = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr_2.sampler(obs=self.obs)
        xx_2 = copula_distr_2.xx

        X_in_p = copula_distr_1.pdf(xx_1)
        X_in_q = copula_distr_2.pdf(xx_1)
        Y_in_p = copula_distr_1.pdf(xx_2)
        Y_in_q = copula_distr_2.pdf(xx_2)
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertTrue(jsd_X_Y > 0 and jsd_X_Y <= 0.2)

    def test_clayton_gumbel(self):
        self.theta = 2
        copula_distr_1 = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr_1.sampler(obs=self.obs)
        xx_1 = copula_distr_1.xx
        self.theta = 5
        self.copula = 'gumbel'
        copula_distr_2 = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        copula_distr_2.sampler(obs=self.obs)
        xx_2 = copula_distr_2.xx

        X_in_p = copula_distr_1.pdf(xx_1)
        X_in_q = copula_distr_2.pdf(xx_1)
        Y_in_p = copula_distr_1.pdf(xx_2)
        Y_in_q = copula_distr_2.pdf(xx_2)
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertTrue(jsd_X_Y > 0 and jsd_X_Y <= 0.3)


if __name__ == '__main__':
    for random_seed in range(5):
        np.random.seed(random_seed)
        random.seed(random_seed)
        unittest.main()
