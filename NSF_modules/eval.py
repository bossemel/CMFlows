import datasets.distributions
import numpy as np
import torch
import datasets
import scipy.stats

from utils import js_divergence_grid


def jsd_eval(args, epoch, model, loader, device, test_dict,
             cm_flow=True):
    """Calculate Jensen-Shannon Divergence of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """

    model.eval()
    with torch.no_grad():
        if cm_flow is True:
            dataset = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=10000)
            data = dataset.xx
            print(data.shape)
            data = torch.tensor(data).float().to(device)
            if args.conditional_copula:
                current_jsd = model.jsd(args=args, inputs=data[:, 0:1],
                                        cond_inputs=data[:, 1:2],
                                        transform_fct=args.transform_fct).sum().item()
            else:
                current_jsd = model.jsd(args=args, inputs=data, transform_fct=args.transform_fct).sum().item()
            if 'jsd_test_copula' in test_dict:
                test_dict["jsd_test_copula"].append(current_jsd)
            else:
                test_dict["jsd_test_copula"] = [current_jsd]
        else:
            dataset = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=10000)
            data = torch.tensor(dataset.xx).float()
            print(data.shape)
            data = data.to(device)
            if args.conditional_copula:
                current_jsd = model.jsd(args=args, inputs=data[:, 0:1],
                                        cond_inputs=data[:, 1:2],
                                        transform_fct=args.transform_fct).sum().item()
            else:
                current_jsd = model.jsd(args=args, inputs=data, transform_fct=args.transform_fct).sum().item()
            if 'jsd_test_copula' in test_dict:
                test_dict["jsd_test_copula"].append(current_jsd)
            else:
                test_dict["jsd_test_copula"] = [current_jsd]

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["jsd_test_copula"])))
    return test_dict


def margin_uniformity(args, epoch, model, cond_inputs=None, transform_fct=None, test_dict=None, num_samples=100000, cm_flow=False):
    """Evaluate Uniformity of best validation model samples.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        test_dict: dictionary with the current epoch stats

    Returns:
        test_dict: updated test_dict
    """
    model.eval()

    with torch.no_grad():
        if args.conditional_copula:
            num_samples = cond_inputs.shape[0]
        current_t_metric_x1, \
            current_m_metric_x1, \
            current_t_metric_x2, \
            current_m_metric_x2 = model.t_metric_eval(num_samples=num_samples, cond_inputs=cond_inputs, transform_fct=transform_fct, cm_flow=cm_flow, device=args.device)
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
             obs=1000, plotname='jsd_test_marginal',
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
        if not cm_flow:
            prob_vector_X = np.exp(model._forward(torch.tensor(grid).to(args.device).float()).cpu().numpy())
        else:
            if marginal_num == '1':
                prob_vector_X = np.exp(model._forward(torch.tensor(grid).to(args.device).float()).cpu().numpy())
            elif marginal_num == '2':
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
