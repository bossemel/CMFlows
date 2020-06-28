import datasets.distributions
import numpy as np
import torch
import scipy
from RealNVP_modules.utils import plot_3D
import matplotlib.pyplot as plt
import os
import seaborn as sns

eps = 0.0001


def jsd_eval(args, epoch, model, loader, device, test_dict,
             transform_model_1=None, transform_model_2=None, transform_inputs=True,
             cm_flow=True):
    """Calculate Jensen-Shannon Divergence of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """

    model.eval()

    if cm_flow is True:
        dataset = datasets.distributions.Copula_Distr(args, transform=False)
        data = dataset.tst.x
        with torch.no_grad():
            current_jsd = model.jsd(args=args, inputs=data, transform_fct=args.transform_fct, cm_flow=cm_flow).sum().item()
        if 'jsd_test_copula' in test_dict:
            test_dict["jsd_test_copula"].append(current_jsd)
        else:
            test_dict["jsd_test_copula"] = [current_jsd]
    else:
        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            with torch.no_grad():
                current_jsd = model.jsd(args=args, inputs=data.detach(), transform_fct=args.transform_fct, cm_flow=cm_flow).sum().item()
            if 'jsd_test_copula' in test_dict:
                test_dict["jsd_test_copula"].append(current_jsd)
            else:
                test_dict["jsd_test_copula"] = [current_jsd]

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["jsd_test_copula"])))
    return test_dict


def margin_uniformity(epoch, model, loader, device, transform_fct, test_dict, num_samples, cm_flow):
    """Evaluate Uniformity of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """
    model.eval()

    with torch.no_grad():
        current_t_metric_x1, \
            current_m_metric_x1, \
            current_t_metric_x2, \
            current_m_metric_x2 = model.t_metric_eval(num_samples=num_samples, transform_fct=transform_fct, cm_flow=cm_flow)
    if 't_1' in test_dict:
        test_dict["t_1"].append(current_t_metric_x1 / num_samples)
        test_dict["m_1"].append(current_m_metric_x1 / num_samples)
        test_dict["t_2"].append(current_t_metric_x2 / num_samples)
        test_dict["m_2"].append(current_m_metric_x2 / num_samples)
    else:
        test_dict["t_1"] = [current_t_metric_x1 / num_samples]
        test_dict["m_1"] = [current_m_metric_x1 / num_samples]
        test_dict["t_2"] = [current_t_metric_x2 / num_samples]
        test_dict["m_2"] = [current_m_metric_x2 / num_samples]

    print('T metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["t_1"])))
    print('M metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["m_1"])))
    print('T metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["t_2"])))
    print('M metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["m_1"])))

    return test_dict


# def plot_margins(args, epoch, model, test_loader):
#     with torch.no_grad():
#         if args.cuda:
#             samples = model.sample(args.obs).detach().cpu().numpy()
#         else:
#             samples = model.sample(args.obs).detach().numpy()
#     margin_x1 = samples[:, 0]
#     margin_x2 = samples[:, 1]
#     if args.transform_fct == 'sigmoid':
#         margin_x1 = scipy.special.expit(margin_x1)
#         margin_x2 = scipy.special.expit(margin_x2)
#     if args.transform_fct == 'gaussian':
#         norm = scipy.stats.norm()
#         margin_x1 = norm.cdf(margin_x1)
#         margin_x2 = norm.cdf(margin_x2)

#     fig, ax = plt.subplots(nrows=2, ncols=1)

#     sns.distplot(margin_x1, color='blue', kde=False, rug=False, ax=ax[0])
#     ax[1].set_xlabel('margin 1', fontsize=16)
#     ax[0].set_ylabel('frequency', fontsize=16)

#     sns.distplot(margin_x2, color='blue', kde=False, rug=False, ax=ax[1])
#     ax[1].set_xlabel('margin 2', fontsize=16)
#     ax[1].set_ylabel('frequency', fontsize=16)

#     fig.tight_layout()
#     fig.savefig(os.path.join(args.figures_path, str(args.copula) + 'margins.pdf'), dpi=300)


def jsd_graph(args, epoch, model, cm_flow=False):
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
        if cm_flow is True:
            pred = model.sample_copula(9000).detach().cpu().numpy()
        else:
            pred = model.sample(9000).detach().cpu().numpy()
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
