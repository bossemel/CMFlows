import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def visualize1D(marginal, model, epoch, args, rng=(-10, 100),
                sample_from_distr=True, best_val=False, obs=None):

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 2, 1)
    data = marginal.sampler(args, obs=obs)
    ax.hist(data)

    res = obs
    xx = torch.linspace(rng[0], rng[1], res).reshape(-1, 1)

    Z = model.log_density(xx).data.numpy()
    ax = fig.add_subplot(1, 2, 2)

    ax.plot(xx, np.exp(Z))
    if not best_val:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}.pdf'.format(epoch)), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval.pdf'.format(epoch)), bbox_inches='tight')
