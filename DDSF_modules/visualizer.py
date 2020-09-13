import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import datasets
import seaborn as sns


def visualize1D(model, epoch, args, best_val=False, obs=10000, name=''):
    """Visualizes the true and predicted marginals.

    Params:
        marginal: kind of marginal distribution
        model: DDSF model
        epoch: epoch to use
        args: passed input arguments
        best_val: whether epoch is best validation epoch
        obs: number of observations to samples
    """
    with torch.no_grad():
        marginal_distr = datasets.distributions.Marginals(args)

        fig = plt.figure(figsize=(8, 6))

        data = marginal_distr.sampler(args=args, obs=obs)
        sns.distplot(data, bins=100, kde=False, label='Test Samples', norm_hist=True, color='orange')

        res = obs
        xx = torch.linspace(np.min(data), np.max(data), res).reshape(-1, 1)

        Z = model.log_density(xx).data.detach().cpu().numpy()

        plt.plot(xx.numpy(), np.exp(Z), label='Predicted PDF', color='royalblue', linewidth=3.0)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('Probability', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        fig.legend(bbox_to_anchor=(0, 0, 0.97, 0.97), fontsize=20)
        fig.tight_layout()

        if not best_val:
            fig.savefig(os.path.join(args.figures_path, 'epoch_{}_{}.pdf'.format(epoch, name)), dpi=300, bbox_inches='tight')
        else:
            fig.savefig(os.path.join(args.figures_path, 'epoch_{}_{}_bestval.pdf'.format(epoch, name)), dpi=300, bbox_inches='tight')
