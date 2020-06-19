import numpy as np
import matplotlib.pyplot as plt
import torch
import os
plt.style.use('ggplot')


def visualize1D(marginal, model, epoch, args,
                sample_from_distr=True, best_val=False, obs=None):

    fig = plt.figure(figsize=(8, 6))

    data = marginal.sampler(args, obs=obs)
    plt.hist(data, bins=100, label='input samples', density=True)

    res = obs
    xx = torch.linspace(np.min(data), np.max(data), res).reshape(-1, 1)

    Z = model.log_density(xx).data.numpy()

    plt.plot(xx.numpy(), np.exp(Z), label='Marginal Flow PDF')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    fig.legend()
    fig.tight_layout()

    if not best_val:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}.pdf'.format(epoch)), dpi=300)
    else:
        fig.savefig(os.path.join(args.figures_path, 'epoch_{}_bestval.pdf'.format(epoch)), dpi=300)
