import torch
import datasets.distributions
import numpy as np
import scipy.spatial
from utils import flow_density, empty_logdets_context, js_divergence_grid
import os


def load_data(args):
    """Data Loader

    Params:
        args: args passed by Training Options

    Returns:
        dataset: full dataset
        num_inputs: dimensions of data
        data_loaders: dictionary containing train, val and test set loader
    """
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    train = torch.load(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_trn'.format(args.copula, args.marginal_1, args.marginal_2)))
    val = torch.load(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_val'.format(args.copula, args.marginal_1, args.marginal_2)))
    test = torch.load(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_tst'.format(args.copula, args.marginal_1, args.marginal_2)))

    # dataset = datasets.distributions.Joint_Distr(args.copula, args.marginal_1, args.marginal_2, args.theta, args.obs,
    #                                              mu=args.mu, var=args.var, alpha=args.alpha, random_seed=args.random_seed)

    train = torch.from_numpy(train)
    train_dataset = torch.utils.data.TensorDataset(train)

    val = torch.from_numpy(val)
    valid_dataset = torch.utils.data.TensorDataset(val)

    test = torch.from_numpy(test)
    test_dataset = torch.utils.data.TensorDataset(test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs)

    data_loaders = {'train_loader': train_loader,
                    'valid_loader': valid_loader,
                    'test_loader': test_loader}
    return (train, val, test), data_loaders


def jsd_eval_marginal_cm(marginal_1, marginal_2, args, model, test_dict,
                         obs=1000, plotname='jsd_test_marginal'):
    """Evaluates the pointwise JSD of the predicted marginal and the true marginal distribution.

    Params:
        marginal_1: distribution of first marginal
        marginal_2: distribution of second marginal
        args: passed arguments
        model: trained CM Flow
        test_dict: dictionary with test results
        obs: number of observations to test the divergence on
        plotname

    Returns:
        test_dict: updated test dictionary
    """
    with torch.no_grad():
        # Get distributions
        args.marginal = marginal_1
        marginal_distr_1 = datasets.distributions.Marginals(marginal_1, obs, mu=args.mu, var=args.var, alpha=args.alpha, low=args.low, high=args.high)
        args.marginal = marginal_2
        marginal_distr_2 = datasets.distributions.Marginals(marginal_2, obs, mu=args.mu, var=args.var, alpha=args.alpha, low=args.low, high=args.high)
        samples_1 = marginal_distr_1.sampler(obs=obs)
        samples_2 = marginal_distr_2.sampler(obs=obs)

        # Get Grid
        grid_1 = np.linspace(np.min(samples_1), np.max(samples_1), obs).reshape(-1, 1)
        grid_2 = np.linspace(np.min(samples_1), np.max(samples_1), obs).reshape(-1, 1)

        # Prob vector pred
        logdets, context = empty_logdets_context(grid_1, args.device)

        output_DDSF_1, logdets_DDSF_1, __ = model.model_DDSF_1.forward((torch.tensor(grid_1).float(), logdets, context))
        output_DDSF_2, logdets_DDSF_2, __ = model.model_DDSF_2.forward((torch.tensor(grid_2).float(), logdets, context))

        args.obs = obs
        prob_vector_X_1 = np.exp(flow_density(output_DDSF_1, logdets_DDSF_1).float().cpu().numpy())
        prob_vector_X_2 = np.exp(flow_density(output_DDSF_2, logdets_DDSF_2).float().cpu().numpy())

        # Prob vector target
        pred_distr_Y_1 = scipy.stats.gaussian_kde(samples_1.T)
        prob_vector_Y_1 = pred_distr_Y_1(grid_1.T).T
        pred_distr_Y_2 = scipy.stats.gaussian_kde(samples_2.T)
        prob_vector_Y_2 = pred_distr_Y_2(grid_2.T).T

        assert np.min(prob_vector_X_1) >= 0
        assert np.min(prob_vector_X_2) >= 0
        assert np.min(prob_vector_Y_1) >= 0
        assert np.min(prob_vector_Y_2) >= 0

        # Calculate JS Divergence
        divergence_1 = js_divergence_grid(prob_vector_X_1, prob_vector_Y_1)
        divergence_2 = js_divergence_grid(prob_vector_X_2, prob_vector_Y_2)

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
