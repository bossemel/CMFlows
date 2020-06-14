import torch
import numpy as np
import datasets.distributions


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

    assert args.copula in [
        'GAUSSIAN', 'TDISTR', 'CLAYTON', 'FRANK', 'GUMBEL'
    ]

    assert args.marginal in [
        'GAUSSIAN'
    ]

    dataset = datasets.distributions.Marginals(args)
    # @Todo: create more distributions

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
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
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


def log_normal(x, mean, log_var, eps=0.00001):
    c = - 0.5 * np.log(2 * np.pi)
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + c
