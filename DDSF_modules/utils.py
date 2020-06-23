import torch
import datasets.distributions
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from datasets.distributions import Marginals
from sklearn.preprocessing import normalize


def load_data(args):
    """Data Loader

    Params:
        args: args passed by Training Options

    Returns:
        dataset: full dataset
        num_cond_inputs: number of conditional inputs (irrelevant for copulas)
        num_inputs: dimensions of data
        data_loaders: dictionary containing train, val and test set loader
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    dataset = datasets.distributions.Marginals(args)

    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)

    num_inputs = dataset.n_dims

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    data_loaders = {'train_loader': train_loader,
                    'valid_loader': valid_loader,
                    'test_loader': test_loader}
    return dataset, num_inputs, data_loaders


def maximum(x, A_max, axis=-1):
    return x.max(axis)[0]


def summation(x, A_max, axis=-1, sum_op=torch.sum):
    return sum_op(torch.exp(x - A_max), axis)


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    A_max = oper_fct(array=A, oper=maximum, axis=axis, keepdims=True)
    B = torch.log(oper_fct(array=A, oper=summation, A_max=A_max, axis=axis, keepdims=True)) + A_max
    return B


def oper_fct(array, oper, A_max=None, axis=-1, keepdims=False):
    a_oper = oper(array, A_max, axis)
    if keepdims:
        shape = []
        for j, s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_normal(inputs, mean, log_var, device, eps=0.00001):
    c = torch.tensor(- 0.5 * math.log(2 * math.pi)).to(device)
    return - (inputs - mean.to(device)) ** 2 / (2. * torch.exp(log_var).to(device) + eps) - log_var.to(device) / 2. + c


def jsd_eval(marginal, args, epoch, model, test_dict,
             obs=100, cm_flow=None, plotname='jsd_test_marginal'):
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
    # data = Marginals.sampler(args, obs=obs)
    data = datasets.distributions.Marginals(args).xx

    xx = torch.linspace(np.min(data), np.max(data), obs).reshape(-1, 1)
    if cm_flow is not None:
        xv, yv = torch.meshgrid((torch.linspace(np.min(data), np.max(data), obs), torch.linspace(np.min(data), np.max(data), obs)))
        # yy = torch.linspace(np.min(data), np.max(data), obs).reshape(-1, 1)
        grid_vector = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), dim=1)

    if args.marginal == 'gaussian':
        true_pdf = scipy.stats.norm.pdf(xx,
                                        loc=args.mu,
                                        scale=args.var)
    elif args.marginal == 'uniform':
        assert hasattr(args, 'low'), 'Please specify lower bound a for %r distribution' % (args.marginal)
        assert hasattr(args, 'high'), 'Please specify upper bound b for %r distribution' % (args.marginal)

        true_pdf = scipy.stats.uniform.pdf(xx,
                                           low=args.low,
                                           high=args.high)
    elif args.marginal == 'gamma':
        assert args.alpha is not None, 'Please specify %r for %r distribution' % (args.marginal)

        true_pdf = scipy.stats.gamma.pdf(xx,
                                         a=args.alpha)

    elif args.marginal == 'lognormal':
        assert args.loc is not None, 'Please specify shape %r distribution' % (args.marginal)

        true_pdf = scipy.stats.lognorm.pdf(xx,
                                           shape=args.alpha)

    if cm_flow is not None:
        Z = np.exp(model.log_density(grid_vector)[cm_flow].detach().numpy()) # [:, cm_flow].reshape(-1, 1)
        Z = Z.reshape(obs, obs, 1).sum(axis=(1 - cm_flow))
        Z = Z / sum(Z)
        true_pdf = true_pdf / sum(true_pdf)
    else:
        Z = np.exp(model.log_density(xx).data.numpy())

    divergence = scipy.spatial.distance.jensenshannon(true_pdf, np.array(Z))
    if cm_flow is not None:
        jsd_name = plotname + '_' + str(cm_flow)
        if jsd_name in test_dict:
            test_dict[jsd_name].append(divergence)
        else:
            test_dict[jsd_name] = [divergence]
    else:
        if plotname in test_dict:
            test_dict[plotname].append(divergence)
        else:
            test_dict[plotname] = [divergence]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(xx.numpy(), true_pdf, label='True PDF')
    plt.plot(xx.numpy(), Z, label='Marginal Flow PDF')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    fig.legend()
    fig.tight_layout()
    if cm_flow is not None:
        fig.savefig(os.path.join(args.figures_path, 'cmflow_marginal_{}_dim_{}.pdf'.format(epoch, cm_flow)), dpi=300)
    else:
        fig.savefig(os.path.join(args.figures_path, 'ddsf_marginal_compare_{}.pdf'.format(epoch)), dpi=300)
    plt.close()
    return test_dict
