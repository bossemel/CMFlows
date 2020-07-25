import datasets
import numpy as np
import torch
import scipy.stats

from utils import js_divergence_grid


def jsd_eval(marginal, args, model, test_dict,
             obs=1000, plotname='jsd_test_marginal',
             cm_flow=False, marginal_num='1'):
    """Calculate pointwise JS-Divergence for the predicted marginal distribution.

    Params:
        marginal: marginal distribution
        args: passed arguments


    Returns:
    """
    with torch.no_grad():
        # Get distributions
        marginal_distr = datasets.distributions.Marginals(args)
        samples = marginal_distr.sampler(args=args, obs=obs)

        # Get Grid
        grid = np.linspace(np.min(samples), np.max(samples), obs).reshape(-1, 1)

        # Prob vector pred
        args.obs = obs
        if not cm_flow:
            prob_vector_X = np.exp(model.log_density(torch.tensor(grid).float()).cpu().numpy())
        else:
            if marginal_num == '1':
                prob_vector_X = np.exp(model.log_density_DDSF_1(torch.tensor(grid).float()).cpu().numpy())
            elif marginal_num == '2':
                prob_vector_X = np.exp(model.log_density_DDSF_2(torch.tensor(grid).float()).cpu().numpy())

        # Prob vector target
        pred_distr_Y = scipy.stats.gaussian_kde(samples.T)
        prob_vector_Y = pred_distr_Y(grid.T).T

        assert np.min(prob_vector_X) >= 0
        assert np.min(prob_vector_Y) >= 0

        # Calculate JS Divergence
        divergence = js_divergence_grid(prob_vector_X, prob_vector_Y)
        print('JS divergence: ', divergence)

        if cm_flow is not None:
            jsd_name = plotname + '_' + str(cm_flow)
            if jsd_name in test_dict:
                test_dict[jsd_name].append(divergence)
            else:
                test_dict[jsd_name] = [divergence]
        else:
            if plotname in test_dict:
                test_dict[plotname].append(divergence)
            else:
                test_dict[plotname] = [divergence]
        return test_dict
