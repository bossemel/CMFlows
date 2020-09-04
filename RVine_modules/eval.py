import numpy as np
import torch

from utils import js_divergence_grid, make_meshgrid


def jsd_eval(args, dim, data, pv_copula, rvine_copula):
    """Calculate pointwise JS-Divergence for the predicted marginal distribution.

    Params:
        marginal: marginal distribution
        args: passed arguments


    Returns:
    """

    with torch.no_grad():
        # Get Grid
        grid = make_meshgrid(obs=10, dim=dim, low=torch.min(data), high=torch.max(data))

        prob_vector_X = rvine_copula.density(torch.tensor(grid).float()).cpu().numpy()
        # Prob vector target
        prob_vector_Y = pv_copula.pdf(grid)

        assert np.min(prob_vector_X) >= 0
        assert np.min(prob_vector_Y) >= 0

        # Calculate JS Divergence
        divergence = js_divergence_grid(prob_vector_X, prob_vector_Y)
        print('Grid JS divergence: ', divergence)
    return divergence
