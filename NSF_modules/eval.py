import datasets.distributions
import numpy as np
import torch
import datasets
import scipy.stats

from utils import js_divergence_grid


def jsd_eval(args, epoch, model, test_dict):
    """Calculate Jensen-Shannon Divergence of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """
    with torch.no_grad():
        current_jsd = model.jsd(args=args, device=args.device).sum().item()
        test_dict["jsd_test_copula"] = current_jsd

    print('JSD in epoch {}:  {:5f}'.format(epoch, test_dict["jsd_test_copula"]))
    return test_dict


def margin_uniformity(args, epoch, model, test_dict=None, num_samples=10000, cm_flow=False):
    """Evaluate Uniformity of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """
    with torch.no_grad():
        current_t_metric_x1, \
            current_m_metric_x1, \
            current_t_metric_x2, \
            current_m_metric_x2 = model.t_metric_eval(args=args, num_samples=num_samples, cm_flow=cm_flow,
                                                      device=args.device)
    test_dict["t_1"] = current_t_metric_x1 / num_samples
    test_dict["m_1"] = current_m_metric_x1 / num_samples
    test_dict["t_2"] = current_t_metric_x2 / num_samples
    test_dict["m_2"] = current_m_metric_x2 / num_samples

    print('T metric x1 in epoch {}:  {:5f}'.format(epoch, test_dict["t_1"]))
    print('M metric x1 in epoch {}:  {:5f}'.format(epoch, test_dict["m_1"]))
    print('T metric x2 in epoch {}:  {:5f}'.format(epoch, test_dict["t_2"]))
    print('M metric x2 in epoch {}:  {:5f}'.format(epoch, test_dict["m_1"]))

    return test_dict


def jsd_eval_1d(args, model, test_dict,
                obs=1000, plotname='jsd_test_marginal',
                cm_flow=False):
    """Calculate pointwise JS-Divergence for the predicted marginal distribution.

    Params:
        marginal: marginal distribution
        args: passed arguments
        model: used model
        test_dict: test_dict for evaluation metrics
        obs: number of observation to sample from
        plotname: name of the plot
        cm_flow: whether model is part of cm flow
        marginal_num: which marginal is used

    Returns:
        test_dict: test_dict with evaluation metrics
    """
    with torch.no_grad():
        # Get distributions
        marginal_distr = datasets.distributions.Marginals(args.marginal, obs, mu_=args.mu, var_=args.var,
                                                          alpha_=args.alpha, low_=args.low, high_=args.high)
        samples = marginal_distr.sampler()

        # Get Grid
        grid = np.linspace(np.min(samples), np.max(samples), obs).reshape(-1, 1)

        # Prob vector pred
        args.obs = obs
        prob_vector_x = np.exp(model._forward(torch.tensor(grid, device=torch.device(args.device)).float())
                               .cpu().numpy())

        # Prob vector target
        pred_distr_y = scipy.stats.gaussian_kde(samples.T)
        prob_vector_y = pred_distr_y(grid.T).T

        assert np.min(prob_vector_x) >= 0
        assert np.min(prob_vector_y) >= 0

        # Calculate JS Divergence
        divergence = js_divergence_grid(prob_vector_x, prob_vector_y)
        print('JS divergence: ', divergence)

        if cm_flow:
            jsd_name = plotname + '_' + str(cm_flow)
            test_dict[jsd_name] = divergence
        else:
            test_dict[plotname] = divergence
        return test_dict
