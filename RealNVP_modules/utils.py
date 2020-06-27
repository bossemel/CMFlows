import os
import matplotlib.pyplot as plt
import torch
import datasets
import scipy


def plot_3D(figures_path, cop_type, grid1, grid2, value, name):
    """Creates 3D plot.

    Params:
        figures_path: path to save the figure
        cop_type: copula type
        grid1, grid2: grid for plotting
        value: value to plot on grid
        name: plot name
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(grid1.reshape(-1), grid2.reshape(-1), value.reshape(-1), cmap=plt.cm.viridis, linewidth=0.2)
    # plt.title(name)
    ax.set_xlabel('U1', fontsize=16)
    ax.set_ylabel('U2', fontsize=16)
    ax.set_zlabel('density', fontsize=16)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(figures_path, str(cop_type) + name), dpi=300, transparent=True)


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

    dataset = datasets.distributions.Copula_Distr(args)

    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)

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
    return dataset, data_loaders


def save_samples_plot(args, epoch, best_model, dataset, obs=3000):
    """Save sample plots

    Params:
        args: args passed by Training Options
        epoch: best validation epoch so far
        best_model: best model so far
        dataset: full dataset
    """
    best_model.eval()
    with torch.no_grad():
        x_synth = best_model.sample(obs).detach().cpu().numpy()
    if args.transform_fct == 'sigmoid':
        val_x = scipy.special.expit(dataset.val.x)
        x_synth = scipy.special.expit(x_synth)
    if args.transform_fct == 'gaussian':
        norm = scipy.stats.norm()
        val_x = norm.cdf(dataset.val.x)
        x_synth = norm.cdf(x_synth)
    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(val_x[:obs, 0], val_x[:obs, 1], '.')
    if args.copula == 'CLAYTON':
        ax.set_title('Clayton Copula', fontsize=16)
    if args.copula == 'FRANK':
        ax.set_title('Frank Copula', fontsize=16)
    if args.copula == 'GUMBEL':
        ax.set_title('Gumbel Copula', fontsize=16)
    ax.set_xlabel('U1', fontsize=16)
    ax.set_ylabel('U2', fontsize=16)

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
    ax.set_title('Copula Flow', fontsize=16)
    ax.set_xlabel('U1', fontsize=16)
    ax.set_ylabel('U2', fontsize=16)

    fig.tight_layout()
    plt.savefig(os.path.join(args.figures_path, '{}_plot_{:03d}.pdf'.format(args.copula, epoch)), dpi=300)
    plt.close()
