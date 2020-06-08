from datasets.copulas import Copula_sampler, sample_clayton, sample_frank, sample_gumbel
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch


def validate(epoch, model, loader, device, writer, global_step, prefix='Validation'):
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
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


def jsd_eval(args, epoch, model, loader, device, writer, global_step, prefix='Validation'):
    # global global_step, writer

    model.eval()
    js_divergence = 0
    new_cop_samples = Copula_sampler(args)

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
            current_jsd = model.jsd(data, new_cop_samples.val.x).sum().item()
            js_divergence += current_jsd
    js_divergence = js_divergence / len(loader.dataset)
    writer.add_scalar('js_divergence/LL', js_divergence, epoch)

    print('JSD in epoch {}:  {:5f}'.format(epoch, js_divergence))
    return js_divergence


def margin_uniformity(epoch, model, loader, device, writer, global_step, prefix='Validation'):
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
            current_t_metric_x1, current_m_metric_x1, current_t_metric_x2, current_m_metric_x2 = model.t_metric_eval(data)
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
    x1 = np.arange(0, 1, 0.01)
    x2 = np.arange(0, 1, 0.01)
    # @Todo: think about meshgrid again
    # grid1, grid2 = torch.from_numpy((np.array(np.meshgrid(x1, x2)).reshape(x1.shape[0] * x2.shape[0], 2))).float()
    grid1, grid2 = np.meshgrid(x1, x2)
    grid1 = grid1.reshape(x1.shape[0] * x2.shape[0], 1)
    grid2 = grid2.reshape(x1.shape[0] * x2.shape[0], 1)
    grid = torch.from_numpy(np.concatenate([grid1.reshape(-1, 1), grid2.reshape(-1, 1)], axis=1)).float()
    print(grid1.shape, grid2.shape, grid.shape)
    pred, _loss = model.forward(grid)
    if args.dataset == 'CLAYTON':
        uu, xx = sample_clayton(args.obs, args.theta, args.seed, uu=grid1, ww=grid2)
        print(xx.shape)
    if args.dataset == 'FRANK':
        uu, xx = sample_frank(args.obs, args.theta, args.seed, uu=grid1, ww=grid2)
        # xx = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)
        print(type(xx))
    if args.dataset == 'GUMBEL':
        xx = sample_gumbel(args.obs, args.theta, args.seed, uu=x1, ww=x2)
    print(' pred shape', pred.shape)
    print(' xx shape', xx.shape)
    # @Todo: sigmoid on prediction anwenden
    pred_2 = np.array(pred[:, 1].detach()).reshape(-1, 1)
    print(pred_2.shape, xx.shape)
    difference = abs(pred_2 - xx)
    print('difference shape', difference.shape, ' grid1', grid1.shape, 'grid2', grid2.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(grid1.reshape(-1), grid2.reshape(-1), difference.reshape(-1), cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()
    #fig.set_axis_labels('X1', 'X2', fontsize=16)
    #fig.show()
    #fig.savefig('normalnormal', dpi=300)


    # @Todo: Fix this mess.
