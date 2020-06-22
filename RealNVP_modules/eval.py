import datasets.distributions
import numpy as np
import torch
import scipy
from RealNVP_modules.utils import plot_3D
import matplotlib.pyplot as plt
import os
import seaborn as sns


def jsd_eval(args, epoch, model, loader, device, current_epoch_test,
             transform_model_1=None, transform_model_2=None, transform_inputs=True,
             cm_flow=True):
    """Calculate Jensen-Shannon Divergence of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        current_epoch_test: dictionary with the current epoch stats

    Returns:
        current_epoch_test: updated current_epoch_test
    """

    model.eval()

    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
            if transform_inputs is True:
                n = data.shape[0]
                context = torch.FloatTensor(n, 1).zero_().to(device)
                logdets = torch.FloatTensor(n).zero_().to(device)
                data_1, __, __ = transform_model_1((data[:, 0].reshape(-1, 1), logdets, context))
                data_2, __, __ = transform_model_2((data[:, 1].reshape(-1, 1), logdets, context))
                data = torch.cat((data_1, data_2), dim=1)
        data = data.to(device)
        with torch.no_grad():
            current_jsd = model.jsd(inputs=data, transform_fct=args.transform_fct, transform_inputs=transform_inputs).sum().item()
        current_epoch_test["jsd_test_copula"].append(current_jsd)

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["jsd_test_copula"])))
    return current_epoch_test


def margin_uniformity(epoch, model, loader, device, transform_fct, current_epoch_test, cm_flow):
    """Evaluate Uniformity of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        current_epoch_test: dictionary with the current epoch stats

    Returns:
        current_epoch_test: updated current_epoch_test
    """
    model.eval()

    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_t_metric_x1, \
                current_m_metric_x1, \
                current_t_metric_x2, \
                current_m_metric_x2 = model.t_metric_eval(data, transform_fct, cm_flow=cm_flow)
        current_epoch_test["t_1"].append(current_t_metric_x1 / len(data))
        current_epoch_test["m_1"].append(current_m_metric_x1 / len(data))
        current_epoch_test["t_2"].append(current_t_metric_x2 / len(data))
        current_epoch_test["m_2"].append(current_m_metric_x2 / len(data))

    print('T metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["t_1"])))
    print('M metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["m_1"])))
    print('T metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["t_2"])))
    print('M metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["m_1"])))

    return current_epoch_test


def plot_margins(args, epoch, model, test_loader):
    with torch.no_grad():
        if args.cuda:
            samples = model.sample(args.obs).detach().cpu().numpy()
        else:
            samples = model.sample(args.obs).detach().numpy()
    margin_x1 = samples[:, 0]
    margin_x2 = samples[:, 1]
    if args.transform_fct == 'sigmoid':
        margin_x1 = scipy.special.expit(margin_x1)
        margin_x2 = scipy.special.expit(margin_x2)
    if args.transform_fct == 'gaussian':
        norm = scipy.stats.norm()
        margin_x1 = norm.cdf(margin_x1)
        margin_x2 = norm.cdf(margin_x2)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    sns.distplot(margin_x1, color='blue', kde=False, rug=False, ax=ax[0])
    ax[1].set_xlabel('margin 1', fontsize=16)
    ax[0].set_ylabel('frequency', fontsize=16)

    sns.distplot(margin_x2, color='blue', kde=False, rug=False, ax=ax[1])
    ax[1].set_xlabel('margin 2', fontsize=16)
    ax[1].set_ylabel('frequency', fontsize=16)

    fig.tight_layout()
    fig.savefig(os.path.join(args.figures_path, str(args.copula) + 'margins.pdf'), dpi=300)


def jsd_graph(args, epoch, model):
    """Creates point-wise graph of true copula, generated samples, and difference
       between the two.

    Params:
        args: passed training args
        epoch: best validation epoch
        model: best validation model
    """
    x1 = np.linspace(0, 1, 300)
    x2 = np.linspace(0, 1, 300)
    grid1, grid2 = np.meshgrid(x1, x2)
    grid1 = grid1.reshape(x1.shape[0] * x2.shape[0], 1)
    grid2 = grid2.reshape(x1.shape[0] * x2.shape[0], 1)
    grid = torch.from_numpy(np.concatenate([grid1.reshape(-1, 1), grid2.reshape(-1, 1)], axis=1)).float()

    with torch.no_grad():
        if args.cuda:
            pred = model.sample(9000).detach().cpu().numpy()
        else:
            pred = model.sample(9000).detach().numpy()
    if args.transform_fct == 'sigmoid':
        pred = scipy.special.expit(pred)
    if args.transform_fct == 'gaussian':
        norm = scipy.stats.norm()
        pred = norm.cdf(pred)
    pred_pdf = scipy.stats.gaussian_kde(pred.T)
    pred_grid = pred_pdf(grid.T)

    cop_pdf = datasets.distributions.copula_pdf(args.copula, args.theta, uu=grid1, vv=grid2).reshape(-1)

    difference = abs(pred_grid - cop_pdf)
    assert pred.all() >= 0 & pred.all() <= 1
    assert pred_grid.all() >= 0 & pred_grid.all() <= 1
    assert cop_pdf.all() >= 0 & cop_pdf.all() <= 1
    assert difference.all() >= 0 & difference.all() < 2

    nan_indices = np.argwhere(np.isnan(cop_pdf))

    # Gumbel pdf contains 'nan' which must be removed
    grid1 = np.delete(grid1, nan_indices)
    grid2 = np.delete(grid2, nan_indices)
    pred_grid = np.delete(pred_grid, nan_indices)
    cop_pdf = np.delete(cop_pdf, nan_indices)
    difference = np.delete(difference, nan_indices)

    plot_3D(args.figures_path, args.copula, grid1, grid2, cop_pdf, 'cop_pdf.pdf')
    plot_3D(args.figures_path, args.copula, grid1, grid2, pred_grid, 'pred_samples.pdf')
    plot_3D(args.figures_path, args.copula, grid1, grid2, difference, 'difference.pdf')

    # @Todo: implement 2d plots
    # pred_grid = pred_grid.reshape(300, 300)
    # cop_pdf = cop_pdf.reshape(300, 300)
    # difference = abs(pred_grid - cop_pdf)
