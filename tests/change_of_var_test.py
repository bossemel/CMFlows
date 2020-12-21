import scipy.stats
import numpy as np
from utils import js_divergence
from utils import gaussian_change_of_var_ND
import torch
import unittest
import random


class Test_Change_of_Var(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_Change_of_Var, self).__init__(*args, **kwargs)
        # Cuda settings
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

        self.normal_distr_1D = scipy.stats.norm()
        self.normal_distr_2D = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., 0],
                                                                                 [0, 1.]])
        self.normal_distr_3D = scipy.stats.multivariate_normal(mean=[0, 0, 0], cov=[[1., 0, 0],
                                                                                    [0, 1., 0], [0, 0, 1]])

    def test_1D(self):
        normal_samples = self.normal_distr_1D.rvs(1000)
        transformed_samples = self.normal_distr_1D.cdf(normal_samples)
        uniform_distr = scipy.stats.uniform()
        uniform_samples = uniform_distr.rvs(1000)
        # compare jsd of transformed_samples with uniform samples
        X_in_p = gaussian_change_of_var_ND(transformed_samples, self.normal_distr_1D.pdf, device=self.device)
        X_in_q = uniform_distr.pdf(transformed_samples)
        Y_in_p = gaussian_change_of_var_ND(uniform_samples, self.normal_distr_1D.pdf, device=self.device)
        Y_in_q = uniform_distr.pdf(uniform_samples)

        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertAlmostEqual(jsd_X_Y, 0)

    def test_2D(self):
        normal_samples = self.normal_distr_2D.rvs(1000)
        transformed_samples = np.apply_along_axis(self.normal_distr_1D.cdf, 1, normal_samples)
        uniform_distr = scipy.stats.uniform()
        uniform_samples = uniform_distr.rvs((1000, 2))
        # compare jsd of transformed_samples with uniform samples
        X_in_p = gaussian_change_of_var_ND(transformed_samples, self.normal_distr_2D.pdf, device=self.device)
        X_in_q = uniform_distr.pdf(transformed_samples).prod(axis=1)
        Y_in_p = gaussian_change_of_var_ND(uniform_samples, self.normal_distr_2D.pdf, device=self.device)
        Y_in_q = uniform_distr.pdf(uniform_samples).prod(axis=1)

        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertAlmostEqual(jsd_X_Y, 0)

    def test_3D(self):
        normal_samples = self.normal_distr_3D.rvs(1000)
        transformed_samples = np.apply_along_axis(self.normal_distr_1D.cdf, 1, normal_samples)
        uniform_distr = scipy.stats.uniform()
        uniform_samples = uniform_distr.rvs((1000, 3))
        # compare jsd of transformed_samples with uniform samples
        X_in_p = gaussian_change_of_var_ND(transformed_samples, self.normal_distr_3D.pdf, device=self.device)
        X_in_q = uniform_distr.pdf(transformed_samples).prod(axis=1)
        Y_in_p = gaussian_change_of_var_ND(uniform_samples, self.normal_distr_3D.pdf, device=self.device)
        Y_in_q = uniform_distr.pdf(uniform_samples).prod(axis=1)
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertAlmostEqual(jsd_X_Y, 0)


if __name__ == '__main__':
    for random_seed in range(5):
        np.random.seed(random_seed)
        random.seed(random_seed)
        unittest.main()
