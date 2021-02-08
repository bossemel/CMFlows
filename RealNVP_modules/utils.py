import torch
import datasets


def load_data(args):
    """Data Loader

    Params:
        args: args passed by Training Options

    Returns:
        dataset: full dataset
        num_inputs: dimensions of data
        data_loaders: dictionary containing train, val and test set loader
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    dataset = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs, random_seed=args.random_seed)

    train_tensor = torch.from_numpy(dataset.trn)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)

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
