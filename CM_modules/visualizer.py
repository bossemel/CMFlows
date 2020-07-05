import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import datasets
import seaborn as sns
from utils import empty_logdets_context, flow_density
import scipy.stats

def visualize1D_CM(model, epoch, args, best_val=False, obs=10000):
    """Visualizes the true and predicted marginals.

    Params:
        marginal: kind of marginal distribution
        model: DDSF model
        epoch: epoch to use
        args: passed input arguments
        best_val: whether epoch is best validation epoch
        obs: number of observations to samples
    """
    marginal_distr = datasets.distributions.Marginals(args)

    def plotting_fct(grid, true_samples, pred_samples, which_marginal):
        fig = plt.figure(figsize=(8, 6))
        pred_distr_Y = scipy.stats.gaussian_kde(true_samples.T)
        prob_vector_Y = pred_distr_Y(grid.T).T

        sns.distplot(true_samples, bins=100, kde=False, label='Input Samples', norm_hist=True, color='orange')
        plt.plot(grid.numpy(), np.exp(pred_samples), label='Predicted Distribution', color='royalblue', linewidth=3.0)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('Probability', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if args.marginal == 'bimodal_gaussian':
            fig.legend(bbox_to_anchor=(0, 0, 0.445, 0.97), fontsize=12)
        else:
            fig.legend(bbox_to_anchor=(0, 0, 0.97, 0.97), fontsize=12)
        fig.tight_layout()

        if not best_val:
            fig.savefig(os.path.join(args.figures_path, 'epoch_{}_marginal_{}.pdf'.format(epoch, which_marginal)), dpi=300, bbox_inches='tight')
        else:
            fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval_marginal_{}.pdf'.format(epoch, which_marginal)), dpi=300, bbox_inches='tight')

    # Get marginal samples
    args.marginal = args.marginal_1
    true_samples_1 = marginal_distr.sampler(args=args, obs=obs)
    args.marginal = args.marginal_2
    true_samples_2 = marginal_distr.sampler(args=args, obs=obs)

    # Get grid
    res = obs
    grid_1 = torch.linspace(np.min(true_samples_1), np.max(true_samples_1), res).reshape(-1, 1)
    grid_2 = torch.linspace(np.min(true_samples_2), np.max(true_samples_2), res).reshape(-1, 1)

    # Prob vector pred
    pred_density_1 = model.log_density_DDSF_1(grid_1).data.detach().cpu().numpy()
    plotting_fct(grid_1, true_samples_1, pred_density_1, which_marginal='1')
    pred_density_2 = model.log_density_DDSF_2(grid_2).data.detach().cpu().numpy()
    plotting_fct(grid_2, true_samples_2, pred_density_2, which_marginal='2')

