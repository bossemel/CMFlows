import datasets.distributions
import argparse
import numpy as np
import random
from utils import js_divergence
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


def visualize_joint(data, axis_1_name=None, axis_2_name=None):
    """Visualize 2D distribution as a seaborn jointplot.
    """
    if axis_1_name is None:
        axis_1_name = 'X1'
    if axis_2_name is None:
        axis_2_name = 'X2'

    fig = plt.figure()
    fig = sns.jointplot(data[:, 0], data[:, 1], kind='hex', stat_func=None)
    fig.set_axis_labels(axis_1_name, axis_2_name, fontsize=16)
    plt.show()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Flows')
    args.random_seed = 2

    # Set Seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    args.theta = 2
    args.obs = 10000
    args.transform_fct = 'gaussian'
    args.copula = 'clayton'
    copula_distr = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs)
    copula_distr.sampler(obs=10000)
    xx = copula_distr.xx

    cop_pdf_values = copula_distr.pdf(xx)
    print(cop_pdf_values.shape)
    print(cop_pdf_values[:10])

    print('Should be zero:')

    X_in_p = copula_distr.pdf(xx)
    X_in_q = copula_distr.pdf(xx)
    Y_in_p = copula_distr.pdf(xx)
    Y_in_q = copula_distr.pdf(xx)

    print(js_divergence(X_in_p, X_in_q, X_in_p, X_in_q))

    print('Should be small:')
    args.theta = 1.5
    copula_distr_2 = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs)
    copula_distr_2.sampler(obs=10000)
    xx_2 = copula_distr_2.xx

    X_in_p = copula_distr.pdf(xx)
    X_in_q = copula_distr_2.pdf(xx)
    Y_in_p = copula_distr.pdf(xx_2)
    Y_in_q = copula_distr_2.pdf(xx_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    print('Should be bigger:')
    args.theta = 2
    args.copula = 'frank'
    copula_distr_2 = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs)
    copula_distr_2.sampler(obs=10000)
    xx_2 = copula_distr_2.xx

    X_in_p = copula_distr.pdf(xx)
    X_in_q = copula_distr_2.pdf(xx)
    Y_in_p = copula_distr.pdf(xx_2)
    Y_in_q = copula_distr_2.pdf(xx_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    print('Should be bigger:')
    args.theta = 5
    args.copula = 'gumbel'
    copula_distr_2 = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs)
    copula_distr_2.sampler(obs=10000)
    xx_2 = copula_distr_2.xx

    X_in_p = copula_distr.pdf(xx)
    X_in_q = copula_distr_2.pdf(xx)
    Y_in_p = copula_distr.pdf(xx_2)
    Y_in_q = copula_distr_2.pdf(xx_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    pred_distr = scipy.stats.gaussian_kde(xx_2.T)
    xx_2_new = pred_distr.resample(10000)
    print(xx_2_new.shape)
    X_in_p = pred_distr.pdf(xx_2_new).T
    X_in_q = copula_distr_2.pdf(xx_2_new.T)
    Y_in_p = pred_distr.pdf(xx_2.T).T
    Y_in_q = copula_distr_2.pdf(xx_2)
    print('min pred distr', np.min(X_in_p), np.max(X_in_p))
    print('min pred distr', np.min(Y_in_p), np.max(Y_in_p))

    print('min pred distr', np.min(X_in_q), np.max(X_in_q))
    print('min pred distr', np.min(Y_in_q), np.max(Y_in_q))

    print('KDE: should be small: ')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    pred_distr = scipy.stats.gaussian_kde(xx.T)
    X_in_p = pred_distr.pdf(xx.T).T
    X_in_q = copula_distr_2.pdf(xx)
    Y_in_p = pred_distr.pdf(xx_2.T).T
    Y_in_q = copula_distr_2.pdf(xx_2)
    print('min pred distr', np.min(X_in_p), np.max(X_in_p))
    print('min pred distr', np.min(Y_in_p), np.max(Y_in_p))

    print('min pred distr', np.min(X_in_q), np.max(X_in_q))
    print('min pred distr', np.min(Y_in_q), np.max(Y_in_q))

    print('KDE: sould be big: ')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))
