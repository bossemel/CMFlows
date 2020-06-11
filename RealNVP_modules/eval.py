from datasets.copulas import copula_pdf
import numpy as np
from tqdm import tqdm
import torch
import scipy
from RealNVP_modules.utils import plot_3D
import copy
import matplotlib.pyplot as plt
import os
import seaborn as sns


def validate(epoch, model, loader, device,
             current_epoch_losses=None, best_dict=None):
    """Return log probabilities on validation set.

    Params:
        epoch: epoch to validate
        model: model to validate
        loader: whether to use train/val/test set loader
        device: used device
        current_epoch_losses: dictionary with the current epoch losses
        best_dict: dictionary containing the best validation loss, best validation epoch
                   and best model

    Returns:
        current_epoch_losses: updated current_epoch_losses
        best_dict: updated best_dict
    """
    model.eval()

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_loss = -model.log_probs(data).mean().item()
        if current_epoch_losses is not None:
            current_epoch_losses["val_loss"].append(current_loss)  # add current iter loss to val loss list.
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss < best_dict['best_validation_loss']:  # if current epoch's mean val acc is greater than the saved best val acc then
                best_dict['best_validation_loss'] = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                best_dict['best_validation_epoch'] = epoch  # set the experiment-wise best val idx to be the current epoch's idx
                best_dict['best_model'] = copy.deepcopy(model)

        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(current_loss))

    pbar.close()
    return current_epoch_losses, best_dict


def test(epoch, model, loader, device,
         current_epoch_test):
    """Return log probabilities on test set.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        current_epoch_test: dictionary with the current epoch test stats

    Returns:
        current_epoch_test: updated current_epoch_test
    """
    model.eval()

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_loss = -model.log_probs(data).mean().item()
        current_epoch_test["test_loss"].append(current_loss)  # add current iter loss to test loss list.

        pbar.update(data.size(0))
        pbar.set_description('Test, Log likelihood in nats in epoch {}: {:.6f}'.format(epoch, np.mean(current_epoch_test["test_loss"])))

    pbar.close()

    return current_epoch_test


def jsd_eval(args, epoch, model, loader, device, current_epoch_test):
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
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_jsd = model.jsd(inputs=data, transform_fct=args.transform_fct).sum().item()
        current_epoch_test["jsd_test"].append(current_jsd)

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["jsd_test"])))
    return current_epoch_test


def margin_uniformity(epoch, model, loader, device, transform_fct, current_epoch_test):
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
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            current_t_metric_x1, \
                current_m_metric_x1, \
                current_t_metric_x2, \
                current_m_metric_x2 = model.t_metric_eval(data, transform_fct)
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
    fig.savefig(os.path.join(args.figures_path, str(args.dataset) + 'margins.pdf'), dpi=300)


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

    cop_pdf = copula_pdf(args.dataset, args.theta, uu=grid1, vv=grid2).reshape(-1)

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

    plot_3D(args.figures_path, args.dataset, grid1, grid2, cop_pdf, 'cop_pdf.pdf')
    plot_3D(args.figures_path, args.dataset, grid1, grid2, pred_grid, 'pred_samples.pdf')
    plot_3D(args.figures_path, args.dataset, grid1, grid2, difference, 'difference.pdf')

    # @Todo: implement 2d plots
    # pred_grid = pred_grid.reshape(300, 300)
    # cop_pdf = cop_pdf.reshape(300, 300)
    # difference = abs(pred_grid - cop_pdf)
