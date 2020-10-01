import scipy.stats
from utils import js_divergence
import numpy as np
import random
import torch
from RVine_modules.options import TrainOptions
from RVine_modules.utils import gen_mv_copula


def jsd_vinecopula(args, true_rvine, obs=100000):
    """Returns JS-Divergence of the predicted Copula and the true Copula.

    Params:
        args: passsed arguments
        true_rvine: rvine from which the dataset was created
        obs: how many observations to create

    Returns:
        divergence: estimated JS-divergence
    """
    # Samples from both distributinos
    samples_target = true_rvine.simulate(obs)
    samples_target_small_change = samples_target + 100000

    # Estimate Copula distr
    # RealNVP outputs the density directly, but not the transformation to
    # uniform marginals. Thus, an estimation with Gaussian KDE is simpler.
    pred_distr = scipy.stats.gaussian_kde(samples_target_small_change.T)
    true_rvine = scipy.stats.gaussian_kde(samples_target.T)
    # Note, that uniform samples means the transformed samples

    # Prob X in both distributions
    prob_X_in_p = pred_distr.pdf(samples_target_small_change.T).T
    prob_X_in_q = true_rvine.pdf(samples_target_small_change.T).T

    # Prob Y in both distributions
    prob_Y_in_q = true_rvine.pdf(samples_target.T).T
    prob_Y_in_p = pred_distr.pdf(samples_target.T).T

    if np.isnan(np.sum(prob_X_in_q)):
        prob_X_in_p = prob_X_in_p[~np.isnan(prob_X_in_q)]
        prob_Y_in_q = prob_Y_in_q[~np.isnan(prob_X_in_q)]
        prob_Y_in_p = prob_Y_in_p[~np.isnan(prob_X_in_q)]
        prob_X_in_q = prob_X_in_q[~np.isnan(prob_X_in_q)]

    if np.isnan(np.sum(prob_Y_in_q)):
        prob_X_in_p = prob_X_in_p[~np.isnan(prob_Y_in_q)]
        prob_X_in_q = prob_X_in_q[~np.isnan(prob_Y_in_q)]
        prob_Y_in_p = prob_Y_in_p[~np.isnan(prob_Y_in_q)]
        prob_Y_in_q = prob_Y_in_q[~np.isnan(prob_Y_in_q)]

    assert np.min(samples_target) >= 0
    assert np.min(samples_target) >= 0
    assert np.min(prob_X_in_p) >= 0
    assert np.min(prob_X_in_q) >= 0
    assert np.min(prob_Y_in_p) >= 0
    assert np.min(prob_Y_in_q) >= 0, '%r' % (np.min(prob_Y_in_q))

    divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                               prob_X_in_q=prob_X_in_q,
                               prob_Y_in_p=prob_Y_in_p,
                               prob_Y_in_q=prob_Y_in_q)

    print('MC-JSD Vine Copula: {}'.format(divergence))


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.RealNVP_part_of_CM_Flow = True

    # Set Seed
    args.random_seed = 12
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    args.viz_obs = 10000

    dataset_trn, dim, pv_cop = gen_mv_copula(args)
    untransformed_samples = pv_cop.simulate(args.viz_obs)

    jsd_vinecopula(args, pv_cop, obs=10000)
