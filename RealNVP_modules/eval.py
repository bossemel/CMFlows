from datasets.copulas import copula_pdf
import numpy as np
from tqdm import tqdm
import torch
import scipy
from RealNVP_modules.utils import plot_3D
import copy


def validate(epoch, model, loader, device,
             current_epoch_losses=None, best_dict=None,
             prefix='Validation'):
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
    # global global_step, writer

    model.eval()

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
        current_epoch_test["test_loss"].append(current_loss)  # add current iter loss to test loss list.

        pbar.update(data.size(0))
        pbar.set_description('Test, Log likelihood in nats in epoch {}: {:.6f}'.format(epoch, np.mean(current_epoch_test["test_loss"])))

    pbar.close()

    return current_epoch_test


def jsd_eval(args, epoch, model, loader, device, current_epoch_test):
    # global global_step, writer

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
            current_jsd = model.jsd(inputs=data, sigmoid=args.sigmoid).sum().item()
        current_epoch_test["jsd_test"].append(current_jsd)

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["jsd_test"])))
    return current_epoch_test


def margin_uniformity(epoch, model, loader, device, sigmoid, current_epoch_test):
    # global global_step, writer

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
                current_m_metric_x2 = model.t_metric_eval(data, sigmoid)
        current_epoch_test["t_1"].append(current_t_metric_x1)
        current_epoch_test["t_2"].append(current_t_metric_x2)
        current_epoch_test["m_1"].append(current_m_metric_x1)
        current_epoch_test["m_2"].append(current_m_metric_x2)

    print('T metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["t_1"])))
    print('T metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["t_2"])))
    print('M metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["m_1"])))
    print('M metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(current_epoch_test["m_1"])))

    return current_epoch_test


def jsd_graph(args, epoch, model, test_loader):
    x1 = np.linspace(0.01, 1, 300)
    x2 = np.linspace(0.01, 1, 300)
    grid1, grid2 = np.meshgrid(x1, x2)
    grid1 = grid1.reshape(x1.shape[0] * x2.shape[0], 1)
    grid2 = grid2.reshape(x1.shape[0] * x2.shape[0], 1)
    grid = torch.from_numpy(np.concatenate([grid1.reshape(-1, 1), grid2.reshape(-1, 1)], axis=1)).float()

    with torch.no_grad():
        if args.cuda:
            pred = model.sample(9000).detach().cpu().numpy()
        else:
            pred = model.sample(9000).detach().numpy()
    pred = scipy.special.expit(pred)
    pred_pdf = scipy.stats.gaussian_kde(pred.T)
    pred_grid = pred_pdf(grid.T)

    cop_pdf = copula_pdf(args.dataset, args.theta, uu=grid1, vv=grid2).reshape(-1)

    difference = abs(pred_grid - cop_pdf)
    assert pred.all() > 0 & pred.all() < 1
    assert pred_grid.all() > 0 & pred_grid.all() < 1
    assert cop_pdf.all() > 0 & cop_pdf.all() < 1
    assert difference.all() >= 0 & difference.all() < 2

    plot_3D(args.figures_path, args.dataset, grid1, grid2, cop_pdf, 'cop_pdf')
    plot_3D(args.figures_path, args.dataset, grid1, grid2, pred_grid, 'pred_samples')
    plot_3D(args.figures_path, args.dataset, grid1, grid2, difference, 'difference')

    pred_grid = pred_grid.reshape(300, 300)
    cop_pdf = cop_pdf.reshape(300, 300)
    difference = abs(pred_grid - cop_pdf)
