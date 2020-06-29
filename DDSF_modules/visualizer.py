import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import datasets
import seaborn as sns


def visualize1D(model, epoch, args, best_val=False, obs=10000):
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

    fig = plt.figure(figsize=(8, 6))

    data = marginal_distr.sampler(args=args, obs=obs)
    sns.distplot(data, bins=100, kde=False, label='Input Samples', norm_hist=True, color='orange')

    res = obs
    xx = torch.linspace(np.min(data), np.max(data), res).reshape(-1, 1)

    Z = model.log_density(xx).data.detach().cpu().numpy()

    plt.plot(xx.numpy(), np.exp(Z), label='Predicted Distribution', color='royalblue', linewidth=3.0)
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
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}.pdf'.format(epoch)), dpi=300, bbox_inches='tight')
    else:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval.pdf'.format(epoch)), dpi=300, bbox_inches='tight')
