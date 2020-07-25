import numpy as np
import torch
import scipy.stats

from utils import js_divergence_grid, make_meshgrid


def jsd_eval(target, args, model, plotname='jsd_test_rvine'):
    """Calculate pointwise JS-Divergence for the predicted marginal distribution.

    Params:
        marginal: marginal distribution
        args: passed arguments


    Returns:
    """
    # Get distributions
    # marginal_distr = datasets.distributions.Marginals(args)
    # samples = marginal_distr.sampler(args=args, obs=obs)

    with torch.no_grad():
        # Get Grid
        print(target.shape[1])
        grid = make_meshgrid(obs=10, dim=target.shape[1])
        # grid = np.linspace(torch.min(target), torch.max(target), target.shape[0]).reshape(-1, 1)

        # Prob vector pred
        prob_vector_X = model.density(torch.tensor(grid).float()).cpu().numpy()
        raise NotImplementedError
        # Prob vector target
        pred_distr_Y = scipy.stats.gaussian_kde(target.T)
        prob_vector_Y = pred_distr_Y(grid.T).T

        assert np.min(prob_vector_X) >= 0
        assert np.min(prob_vector_Y) >= 0

        # Calculate JS Divergence
        divergence = js_divergence_grid(prob_vector_X, prob_vector_Y)
        print('JS divergence: ', divergence)

        # if cm_flow is not None:
        #     jsd_name = plotname + '_' + str(cm_flow)
        #     if jsd_name in test_dict:
        #         test_dict[jsd_name].append(divergence)
        #     else:
        #         test_dict[jsd_name] = [divergence]
        # else:
        #     if plotname in test_dict:
        #         test_dict[plotname].append(divergence)
        #     else:
        #         test_dict[plotname] = [divergence]
        # return test_dict
