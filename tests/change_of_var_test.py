import scipy.stats
import numpy as np
from utils import gaussian_pdf_log, js_divergence
from utils.visualizer import visualize_joint
import matplotlib.pyplot as plt



def gaussian_change_of_var_2D(inputs, original_pdf):
    assert np.max(inputs) < 1
    assert np.min(inputs) > 0
    normal_distr = scipy.stats.norm()
    recast_inputs = np.apply_along_axis(normal_distr.ppf, 1, inputs)
    output = original_pdf(recast_inputs) / normal_distr.pdf(recast_inputs).prod(axis=1)
    assert not np.isnan(output.sum())
    assert not np.isinf(output.sum())
    assert np.min(output) >= 0
    return output


def gaussian_change_of_var_1D(inputs, original_pdf):
    assert np.max(inputs) < 1
    assert np.min(inputs) > 0
    normal_distr = scipy.stats.norm()
    recast_inputs = normal_distr.ppf(inputs)
    assert not np.isnan(recast_inputs.sum())
    assert not np.isinf(recast_inputs.sum())
    print(recast_inputs.shape)
    output = original_pdf(recast_inputs) / normal_distr.pdf(recast_inputs)
    assert not np.isnan(output.sum())
    assert not np.isinf(output.sum())
    assert np.min(output) >= 0
    return output


def plot_density(pdf):
    grid = np.linspace(0.0001, 0.9999, 100)
    plt.plot(grid, gaussian_change_of_var_1D(grid, pdf))
    plt.show()


if __name__ == '__main__':

    # 1. Transform Normal to uniform
    normal_distr = scipy.stats.norm()
    print(normal_distr)
    normal_samples = normal_distr.rvs(1000)
    transformed_samples = normal_distr.cdf(normal_samples) #should be uniform now

    uniform_distr = scipy.stats.uniform()
    uniform_samples = uniform_distr.rvs(1000)

    # compare jsd of transformed_samples with uniform samples
    X_in_p = gaussian_change_of_var_1D(transformed_samples, normal_distr.pdf)
    X_in_q = uniform_distr.pdf(transformed_samples)
    Y_in_p = gaussian_change_of_var_1D(uniform_samples, normal_distr.pdf)
    Y_in_q = uniform_distr.pdf(uniform_samples)

    print('Should be zero:')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    #2. Transform MN Normal to uniform column wise
    normal_distr_mv = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., 0],
                                                                        [0, 1.]])
    normal_samples = normal_distr_mv.rvs(1000)
    print('normal samples mv', normal_samples.shape)

    transformed_samples = np.apply_along_axis(normal_distr.cdf, 1, normal_samples)
    print('trasnformed samples shape', transformed_samples.shape)
    uniform_distr = scipy.stats.uniform()
    uniform_samples = uniform_distr.rvs((1000, 2))

    #visualize_joint(uniform_samples, 'tests/', 'uniform', axis_1_name=None, axis_2_name=None)
    #visualize_joint(transformed_samples, 'tests/', 'transformed_samples', axis_1_name=None, axis_2_name=None)

    # compare jsd of transformed_samples with uniform samples
    X_in_p = gaussian_change_of_var_2D(transformed_samples, normal_distr_mv.pdf)
    X_in_q = uniform_distr.pdf(transformed_samples).prod(axis=1)
    Y_in_p = gaussian_change_of_var_2D(uniform_samples, normal_distr_mv.pdf)
    Y_in_q = uniform_distr.pdf(uniform_samples).prod(axis=1)

    print('Should be zero:')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # @Todo: think about a test for conditional distribution?
