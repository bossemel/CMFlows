import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns


def visualize1D_CM(marginal, model, epoch, args, rng=(-10, 100),
                   sample_from_distr=True, best_val=False, obs=None):
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 2, 1)
    data = marginal.sampler(args, obs=obs)
    ax.hist(data[:, 0])

    res = obs
    xx = torch.linspace(rng[0], rng[1], res).reshape(-1, 1)

    Z = model.log_density(xx).data.numpy()
    ax = fig.add_subplot(1, 2, 2)

    ax.plot(xx, np.exp(Z))
    if not best_val:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}.pdf'.format(epoch)), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval.pdf'.format(epoch)), bbox_inches='tight')


def visualize_joint(data, args):

    fig = plt.figure()
    fig = sns.jointplot(data.trn.x[:, 0], data.trn.x[:, 1], kind='kde', stat_func=None)
    fig.set_axis_labels('X1', 'X2', fontsize=16)
    fig.savefig(os.path.join(args.figures_path, 'true_distr'), dpi=300, bbox_inches='tight')
