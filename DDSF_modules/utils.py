import torch
import datasets.distributions
import math
import numpy as np
import scipy
from utils import js_divergence


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
             obs=10000, cm_flow=None, plotname='jsd_test_marginal'):
    # Get distributions
    marginal_distr = datasets.distributions.Marginals(args)
    # Samples from both distributinos
    samples_target = marginal_distr.sampler(args=args, obs=obs)
    samples_target = torch.tensor(samples_target.reshape(-1, 1)).float()

    # @Todo: find something for samples_pred
    # Temporary solution: samples from uniform distr
    samples_pred = scipy.stats.uniform.rvs(loc=0, scale=1, size=obs)
    samples_pred = torch.tensor(samples_pred.reshape(-1, 1)).float()

    # Prob X in both distributions
    prob_X_in_p = np.exp(model.log_density(samples_pred).detach().cpu().numpy())
    prob_X_in_q = marginal_distr.pdf(args=args, inputs=samples_pred)

    # Prob Y in both distributions
    prob_Y_in_q = marginal_distr.pdf(args=args, inputs=samples_target)
    prob_Y_in_p = np.exp(model.log_density(samples_target).detach().cpu().numpy())

    divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                               prob_X_in_q=prob_X_in_q,
                               prob_Y_in_p=prob_Y_in_p,
                               prob_Y_in_q=prob_Y_in_q)

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
