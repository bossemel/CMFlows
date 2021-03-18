import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import datasets
import seaborn as sns


def visualize1d(model, epoch, args, best_val=False, obs=1000, name=''):
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
        marginal_distr = datasets.distributions.Marginals(args.marginal, obs, mu_=args.mu, var_=args.var,
                                                          alpha_=args.alpha, low_=args.low, high_=args.high,
                                                          random_seed=args.random_seed)

        fig = plt.figure(figsize=(8, 6))

        data = marginal_distr.sampler()
        sns.distplot(data, bins=100, kde=False, label='Test Samples', norm_hist=True, color='orange')

        xx = torch.linspace(np.min(data), np.max(data), obs).reshape(-1, 1)
        zz = np.exp(model._forward(xx.to(args.device)).data.detach().cpu().numpy())
        plt.plot(xx.numpy(), zz, label='Predicted PDF', color='royalblue', linewidth=3.0)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('Probability', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        fig.legend(bbox_to_anchor=(0, 0, 0.97, 0.97), fontsize=20)
        fig.tight_layout()

        if not best_val:
            fig.savefig(os.path.join(args.figures_path, 'epoch_{}_{}.pdf'.format(epoch, name)),
                        dpi=300, bbox_inches='tight')
        else:
            fig.savefig(os.path.join(args.figures_path, 'epochsubl_{}_bestval.pdf'.format(name)),
                        dpi=300, bbox_inches='tight')
