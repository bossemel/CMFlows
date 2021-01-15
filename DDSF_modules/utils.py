import torch
import datasets.distributions
import math


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

    dataset = datasets.distributions.Marginals(args.marginal, args.obs, mu=args.mu, var=args.var, alpha=args.alpha,
                                               low=args.low, high=args.high, random_seed=args.random_seed)

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
