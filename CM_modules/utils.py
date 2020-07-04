import torch
import datasets
import numpy as np
import scipy.spatial
from utils import flow_density, empty_logdets_context, js_divergence_grid


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

    # @Todo: change thiso.
    dataset = datasets.distributions.Joint_Distr(args)

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
    return dataset, data_loaders, train_tensor


def jsd_eval_marginal_cm(marginal_1, marginal_2, args, epoch, model, test_dict,
                         obs=1000, plotname='jsd_test_marginal'):
    # Get distributions
    args.marginal = marginal_1
    marginal_distr_1 = datasets.distributions.Marginals(args)
    samples = marginal_distr_1.sampler(args=args, obs=obs)

    # Get Grid
    grid = np.linspace(np.min(samples), np.max(samples), obs).reshape(-1, 1)

    # Prob vector pred
    logdets, context = empty_logdets_context(grid, args.device)

    output_DDSF_1, logdets_DDSF_1, __ = model.model_DDSF_1.forward((torch.tensor(grid).float(), logdets, context))
    output_DDSF_2, logdets_DDSF_2, __ = model.model_DDSF_2.forward((torch.tensor(grid).float(), logdets, context))

    args.obs = obs
    prob_vector_X_1 = np.exp(flow_density(output_DDSF_1, logdets_DDSF_1).float().detach().cpu().numpy())
    prob_vector_X_2 = np.exp(flow_density(output_DDSF_2, logdets_DDSF_2).float().detach().cpu().numpy())

    # Prob vector target
    pred_distr_Y = scipy.stats.gaussian_kde(samples.T)
    prob_vector_Y = pred_distr_Y(grid.T).T

    assert np.min(prob_vector_X_1) >= 0
    assert np.min(prob_vector_X_2) >= 0
    assert np.min(prob_vector_Y) >= 0

    # Calculate JS Divergence
    divergence_1 = js_divergence_grid(prob_vector_X_1, prob_vector_Y)
    divergence_2 = js_divergence_grid(prob_vector_X_2, prob_vector_Y)

    print('Marginal 1 Divergence: ', divergence_1)
    print('Marginal 2 Divergence: ', divergence_2)

    jsd_name = plotname + '_' + str(0)
    if jsd_name in test_dict:
        test_dict[jsd_name].append(divergence_1)
    else:
        test_dict[jsd_name] = [divergence_1]

    jsd_name = plotname + '_' + str(1)
    if jsd_name in test_dict:
        test_dict[jsd_name].append(divergence_2)
    else:
        test_dict[jsd_name] = [divergence_2]

    return test_dict
