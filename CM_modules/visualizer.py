import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import scipy.special
from datasets.distributions import Copula_Distr
import datasets
import seaborn as sns

# def visualize1D_CM(marginal, model, epoch, args, rng=(-10, 100),
#                    sample_from_distr=True, best_val=False, obs=None):
#     fig = plt.figure(figsize=(8, 6))

#     ax = fig.add_subplot(1, 2, 1)
#     data = marginal.sampler(args, obs=obs)
#     ax.hist(data[:, 0])

#     res = obs
#     xx = torch.linspace(rng[0], rng[1], res).reshape(-1, 1)

#     Z = model.log_density(xx).data.numpy()
#     ax = fig.add_subplot(1, 2, 2)

#     ax.plot(xx, np.exp(Z))
#     if not best_val:
#         fig.savefig(os.path.join(args.figures_path, 'epoch_{}.pdf'.format(epoch)), bbox_inches='tight')
#     else:
#         fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval.pdf'.format(epoch)), bbox_inches='tight')


# def save_samples_plot_copula(args, epoch, model, dataset, obs=3000):
#     """Save sample plots

#     Params:
#         args: args passed by Training Options
#         epoch: best validation epoch so far
#         best_model: best model so far
#         dataset: full dataset
#     """
#     copula_xx = Copula_Distr.sampler(args, transform=False, obs=obs)
#     fig = plt.figure()

#     ax = fig.add_subplot(121)
#     ax.plot(copula_xx[:obs, 0], copula_xx[:obs, 1], '.')

#     model.eval()
#     with torch.no_grad():
#         x_synth = model.sample(obs).detach().cpu().numpy()
#     if args.transform_fct == 'sigmoid':
#         x_synth = scipy.special.expit(x_synth)
#     if args.transform_fct == 'gaussian':
#         norm = scipy.stats.norm()
#         x_synth = norm.cdf(x_synth)

#     if args.copula == 'CLAYTON':
#         ax.set_title('Clayton Copula', fontsize=16)
#     if args.copula == 'FRANK':
#         ax.set_title('Frank Copula', fontsize=16)
#     if args.copula == 'GUMBEL':
#         ax.set_title('Gumbel Copula', fontsize=16)
#     ax.set_xlabel('U1', fontsize=16)
#     ax.set_ylabel('U2', fontsize=16)

#     ax = fig.add_subplot(122)
#     ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
#     ax.set_title('Copula Flow', fontsize=16)
#     ax.set_xlabel('U1', fontsize=16)
#     ax.set_ylabel('U2', fontsize=16)

#     fig.tight_layout()
#     plt.savefig(os.path.join(args.figures_path, '{}_plot_{:03d}.pdf'.format(args.copula, epoch)), dpi=300)
#     plt.close()

