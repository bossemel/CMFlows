from utils import js_divergence
import argparse
import pyvinecopulib as pv
from datasets.distributions import marginal_transform
from utils import normalize
import numpy as np
import torch


def gen_mv_copula(args, theta):
    if args.mix is False:
        if args.copula == 'clayton':
            pair_copula = pv.BicopFamily.clayton
        elif args.copula == 'frank':
            pair_copula = pv.BicopFamily.frank
        elif args.copula == 'gumbel':
            pair_copula = pv.BicopFamily.gumbel

        # Specify pair-copulas
        bicop = pv.Bicop(family=pair_copula, parameters=[theta])
        pcs = [[bicop, bicop, bicop], [bicop, bicop], [bicop]]
    else:
        bicop_1 = pv.Bicop(family=pv.BicopFamily.clayton, parameters=[2])
        bicop_2 = pv.Bicop(family=pv.BicopFamily.frank, parameters=[5])
        bicop_3 = pv.Bicop(family=pv.BicopFamily.gumbel, parameters=[5])
        pcs = [[bicop_1, bicop_2, bicop_3], [bicop_1, bicop_2], [bicop_3]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 0, 0], [4, 0, 0, 0]])

    # Set-up a vine copula
    copula = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    copula_samples = copula.simulate(n=args.obs)
    if not args.disable_marginal:
        for dim in range(copula_samples.shape[1]):
            copula_samples[:, dim] = normalize(marginal_transform(copula_samples[:, dim], marginal=args.marginal, mu=args.mu, var=args.var, alpha=args.alpha))
    assert not np.isnan(np.sum(copula_samples)), '{}'.format(copula_samples[np.isnan(copula_samples)])
    return torch.from_numpy(copula_samples), copula_samples.shape[1], copula


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Flows')
    args.random_seed = 2
    args.mix = False
    args.copula = 'clayton'
    args.obs = 10000
    args.disable_marginal = True
    args.viz_obs = 100000

    # Should be zero:
    print('Should be zero: ')
    __, __, pv_cop = gen_mv_copula(args, theta=2)
    untransformed_samples = pv_cop.simulate(args.viz_obs)

    X_in_p = pv_cop.pdf(untransformed_samples)
    X_in_q = pv_cop.pdf(untransformed_samples)
    Y_in_p = pv_cop.pdf(untransformed_samples)
    Y_in_q = pv_cop.pdf(untransformed_samples)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # Should be small:
    print('Should be medium: ')
    args.copula = 'gumbel'
    __, __, pv_cop_2 = gen_mv_copula(args, theta=2)
    untransformed_samples_2 = pv_cop_2.simulate(args.viz_obs)

    X_in_p = pv_cop.pdf(untransformed_samples)
    X_in_q = pv_cop_2.pdf(untransformed_samples)
    Y_in_p = pv_cop.pdf(untransformed_samples_2)
    Y_in_q = pv_cop_2.pdf(untransformed_samples_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # Should be small:
    print('Should be small: ')
    args.copula = 'clayton'
    __, __, pv_cop_2 = gen_mv_copula(args, theta=3)
    untransformed_samples_2 = pv_cop_2.simulate(args.viz_obs)

    X_in_p = pv_cop.pdf(untransformed_samples)
    X_in_q = pv_cop_2.pdf(untransformed_samples)
    Y_in_p = pv_cop.pdf(untransformed_samples_2)
    Y_in_q = pv_cop_2.pdf(untransformed_samples_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # Should be bigger:
    print('Should be bigger: ')
    args.copula = 'clayton'
    __, __, pv_cop_2 = gen_mv_copula(args, theta=27)
    untransformed_samples_2 = pv_cop_2.simulate(args.viz_obs)

    X_in_p = pv_cop.pdf(untransformed_samples)
    X_in_q = pv_cop_2.pdf(untransformed_samples)
    Y_in_p = pv_cop.pdf(untransformed_samples_2)
    Y_in_q = pv_cop_2.pdf(untransformed_samples_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # Should be bigger:
    print('Should be bigger: ')
    args.copula = 'clayton'
    __, __, pv_cop_2 = gen_mv_copula(args, theta=27)
    untransformed_samples_2 = pv_cop_2.simulate(args.viz_obs)

    X_in_p = pv_cop.pdf(untransformed_samples)
    X_in_q = pv_cop_2.pdf(untransformed_samples)
    Y_in_p = pv_cop.pdf(untransformed_samples_2)
    Y_in_q = pv_cop_2.pdf(untransformed_samples_2)

    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))
