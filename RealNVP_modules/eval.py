from datasets.copulas import copula_pdf, sample_clayton
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import scipy
from RealNVP_modules.utils import plot_3D


def validate(epoch, model, loader, device, writer, global_step, current_epoch_losses=None, prefix='Validation'):
    # global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
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
            current_loss = -model.log_probs(data, cond_data).mean().item()
            val_loss += current_loss  # sum up batch loss
        if current_epoch_losses is not None:
            current_epoch_losses["val_loss"].append(current_loss)  # add current iter loss to val loss list.
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss, epoch)

    pbar.close()
    return val_loss / len(loader.dataset), current_epoch_losses


def jsd_eval(args, epoch, model, loader, device, writer, global_step, prefix='Validation'):
    # global global_step, writer

    model.eval()
    js_divergence = 0

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
            current_jsd = model.jsd(data, cond_data, args.sigmoid).sum().item()
            js_divergence += current_jsd
    js_divergence = js_divergence / len(loader.dataset)
    writer.add_scalar('js_divergence/LL', js_divergence, epoch)

    print('JSD in epoch {}:  {:5f}'.format(epoch, js_divergence))
    return js_divergence


def margin_uniformity(epoch, model, loader, device, writer, global_step, sigmoid, prefix='Validation'):
    # global global_step, writer

    model.eval()
    t_metric_x1 = 0
    t_metric_x2 = 0
    m_metric_x1 = 0
    m_metric_x2 = 0

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
            current_t_metric_x1, current_m_metric_x1, current_t_metric_x2, current_m_metric_x2 = model.t_metric_eval(data, sigmoid)
            t_metric_x1 += current_t_metric_x1
            t_metric_x2 += current_t_metric_x2
            m_metric_x1 += current_m_metric_x1
            m_metric_x2 += current_m_metric_x2

    t_metric_x1 = t_metric_x1 / len(loader.dataset)
    t_metric_x2 = t_metric_x2 / len(loader.dataset)
    m_metric_x1 = m_metric_x1 / len(loader.dataset)
    m_metric_x2 = m_metric_x2 / len(loader.dataset)

    writer.add_scalar('t_metric_x1/LL', t_metric_x1, epoch)
    writer.add_scalar('t_metric_x2/LL', t_metric_x2, epoch)
    writer.add_scalar('m_metric_x1/LL', m_metric_x1, epoch)
    writer.add_scalar('m_metric_x2/LL', m_metric_x2, epoch)

    print('T metric x1 in epoch {}:  {:5f}'.format(epoch, t_metric_x1))
    print('T metric x2 in epoch {}:  {:5f}'.format(epoch, t_metric_x2))
    print('M metric x1 in epoch {}:  {:5f}'.format(epoch, m_metric_x1))
    print('M metric x2 in epoch {}:  {:5f}'.format(epoch, m_metric_x2))

    return t_metric_x1, t_metric_x1, m_metric_x1, m_metric_x2


def jsd_graph(args, epoch, model, test_loader, writer, global_step, prefix='Test'):
    x1 = np.arange(0.1, 1, 0.01)
    x2 = np.arange(0.1, 1, 0.01)
    grid1, grid2 = np.meshgrid(x1, x2)
    grid1 = grid1.reshape(x1.shape[0] * x2.shape[0], 1)
    grid2 = grid2.reshape(x1.shape[0] * x2.shape[0], 1)
    grid = torch.from_numpy(np.concatenate([grid1.reshape(-1, 1), grid2.reshape(-1, 1)], axis=1)).float()

    with torch.no_grad():
        pred, _loss = model.forward(grid)
    #pred = scipy.special.expit(pred)
    pred_pdf = scipy.stats.gaussian_kde(pred.T)
    pred_grid = pred_pdf(grid.T)

    cop_pdf = copula_pdf(args.dataset, args.theta, uu=grid1, vv=grid2).reshape(-1)

    difference = abs(pred_grid - cop_pdf)

    plot_3D(args.figures_path, args.dataset, grid1, grid2, cop_pdf, 'cop_pdf')
    plot_3D(args.figures_path, args.dataset, grid1, grid2, pred_grid, 'pred_samples')
    plot_3D(args.figures_path, args.dataset, grid1, grid2, difference, 'difference')
