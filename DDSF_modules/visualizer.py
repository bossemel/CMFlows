import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import datasets


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
    plt.hist(data, bins=100, label='Input Samples', density=True)

    res = obs
    xx = torch.linspace(np.min(data), np.max(data), res).reshape(-1, 1)

    Z = model.log_density(xx).data.detach().cpu().numpy()

    plt.plot(xx.numpy(), np.exp(Z), label='Predicted Distribution')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    fig.legend(bbox_to_anchor=(0, 0, 0.97, 0.97), fontsize=12)
    fig.tight_layout()

    if not best_val:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}.pdf'.format(epoch)), dpi=300)
    else:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval.pdf'.format(epoch)), dpi=300)


# def marginal_plots(marginal, args, epoch, model, test_dict,
#                    obs=100, cm_flow=None, plotname='jsd_test_marginal'):
#     """Calculate Jensen-Shannon Divergence of best validation model samples.

#     Params:
#         epoch: best validation epoch
#         model: best validation model
#         loader: whether to use train/val/test set loader
#         device: used device
#         test_dict: dictionary with the current epoch stats

#     Returns:
#         test_dict: updated test_dict
#     """
#     # data = Marginals.sampler(args, obs=obs)
#     marginal_distr = datasets.distributions.Marginals(args)

#     data = datasets.distributions.Marginals(args).xx

#     xx = torch.linspace(np.min(data), np.max(data), obs).reshape(-1, 1)
#     if cm_flow is not None:
#         xv, yv = torch.meshgrid((torch.linspace(np.min(data), np.max(data), obs), torch.linspace(np.min(data), np.max(data), obs)))
#         # yy = torch.linspace(np.min(data), np.max(data), obs).reshape(-1, 1)
#         grid_vector = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), dim=1)

#     true_pdf = marginal_distr.pdf(args=args, inputs=xx)

#     if cm_flow is not None:
#         Z = np.exp(model.log_density(grid_vector)[cm_flow].detach().numpy()) # [:, cm_flow].reshape(-1, 1)
#         Z = Z.reshape(obs, obs, 1).sum(axis=(1 - cm_flow))
#         Z = Z / sum(Z)
#         true_pdf = true_pdf / sum(true_pdf)
#     else:
#         if args.cuda:
#             Z = np.exp(model.log_density(xx).data.detach().cpu().numpy())
#         else:
#             Z = np.exp(model.log_density(xx).data.numpy())

#     fig = plt.figure(figsize=(8, 6))
#     plt.plot(xx.numpy(), true_pdf, label='True PDF')
#     plt.plot(xx.numpy(), Z, label='Marginal Flow PDF')
#     plt.xlabel('x', fontsize=16)
#     plt.ylabel('Probability', fontsize=16)
#     fig.legend()
#     fig.tight_layout()
#     if cm_flow is not None:
#         fig.savefig(os.path.join(args.figures_path, 'cmflow_marginal_{}_dim_{}.pdf'.format(epoch, cm_flow)), dpi=300)
#     else:
#         fig.savefig(os.path.join(args.figures_path, 'ddsf_marginal_compare_{}.pdf'.format(epoch)), dpi=300)
#     plt.close()
#     return test_dict
