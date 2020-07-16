import torch
import datasets.distributions
import math
import numpy as np
from utils import js_divergence_grid
import scipy.stats


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

    train_tensor = torch.from_numpy(dataset.trn)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False, **kwargs)

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
    return dataset, data_loaders


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


def jsd_eval(marginal, args, model, test_dict,
             obs=1000, plotname='jsd_test_marginal',
             cm_flow=False, marginal_num='1'):
    """Calculate pointwise JS-Divergence for the predicted marginal distribution.

    Params:
        marginal: marginal distribution
        args: passed arguments


    Returns:
    """
    # Get distributions
    marginal_distr = datasets.distributions.Marginals(args)
    samples = marginal_distr.sampler(args=args, obs=obs)

    # Get Grid
    grid = np.linspace(np.min(samples), np.max(samples), obs).reshape(-1, 1)

    # Prob vector pred
    args.obs = obs
    if not cm_flow:
        prob_vector_X = np.exp(model.log_density(torch.tensor(grid).float()).detach().cpu().numpy())
    else:
        if marginal_num == '1':
            prob_vector_X = np.exp(model.log_density_DDSF_1(torch.tensor(grid).float()).detach().cpu().numpy())
        elif marginal_num == '2':
            prob_vector_X = np.exp(model.log_density_DDSF_2(torch.tensor(grid).float()).detach().cpu().numpy())

    # Prob vector target
    pred_distr_Y = scipy.stats.gaussian_kde(samples.T)
    prob_vector_Y = pred_distr_Y(grid.T).T

    assert np.min(prob_vector_X) >= 0
    assert np.min(prob_vector_Y) >= 0

    # Calculate JS Divergence
    divergence = js_divergence_grid(prob_vector_X, prob_vector_Y)
    print('JS divergence: ', divergence)

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
    return test_dict
