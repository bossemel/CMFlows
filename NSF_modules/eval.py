import datasets.distributions
import numpy as np
import torch
import datasets
import scipy.stats

from utils import js_divergence_grid


def jsd_eval(args, epoch, model, device, test_dict):
    """Calculate Jensen-Shannon Divergence of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """

    model.eval()
    with torch.no_grad():
        current_jsd = model.jsd(args=args, device=args.device).sum().item()
        if 'jsd_test_copula' in test_dict:
            test_dict["jsd_test_copula"].append(current_jsd)
        else:
            test_dict["jsd_test_copula"] = [current_jsd]

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["jsd_test_copula"])))
    return test_dict


def margin_uniformity(args, epoch, model, transform_fct=None, test_dict=None, num_samples=10000, cm_flow=False):
    """Evaluate Uniformity of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """
    model.eval()

    with torch.no_grad():
        current_t_metric_x1, \
            current_m_metric_x1, \
            current_t_metric_x2, \
            current_m_metric_x2 = model.t_metric_eval(args=args, num_samples=num_samples, transform_fct=transform_fct, cm_flow=cm_flow, device=args.device)
    if 't_1' in test_dict:
        test_dict["t_1"].append(current_t_metric_x1 / num_samples)
        test_dict["m_1"].append(current_m_metric_x1 / num_samples)
        test_dict["t_2"].append(current_t_metric_x2 / num_samples)
        test_dict["m_2"].append(current_m_metric_x2 / num_samples)
    else:
        test_dict["t_1"] = [current_t_metric_x1 / num_samples]
        test_dict["m_1"] = [current_m_metric_x1 / num_samples]
        test_dict["t_2"] = [current_t_metric_x2 / num_samples]
        test_dict["m_2"] = [current_m_metric_x2 / num_samples]

    print('T metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["t_1"])))
    print('M metric x1 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["m_1"])))
    print('T metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["t_2"])))
    print('M metric x2 in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["m_1"])))

    return test_dict


def jsd_eval_1D(marginal, args, model, test_dict,
             obs=10000, plotname='jsd_test_marginal',
             cm_flow=False, marginal_num='1'):
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
        marginal_distr = datasets.distributions.Marginals(args.marginal, obs, mu=args.mu, var=args.var, alpha=args.alpha, low=args.low, high=args.high)
        samples = marginal_distr.sampler(obs=obs)

        # Get Grid
        grid = np.linspace(np.min(samples), np.max(samples), obs).reshape(-1, 1)

        # Prob vector pred
        args.obs = obs
        prob_vector_X = np.exp(model._forward(torch.tensor(grid).to(args.device).float()).cpu().numpy())

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
