import datasets.distributions
import numpy as np
import torch


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
        current_jsd = model.jsd(args=args, transform_fct=args.transform_fct, cm_flow=cm_flow, device=args.device).sum().item()

        if 'jsd_test_copula' in test_dict:
            test_dict["jsd_test_copula"].append(current_jsd)
        else:
            test_dict["jsd_test_copula"] = [current_jsd]

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["jsd_test_copula"])))
    return test_dict


def margin_uniformity(args, epoch, model, context=None, transform_fct=None, test_dict=None, num_samples=100000, cm_flow=False):
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
            num_samples = context.shape[0]
        current_t_metric_x1, \
            current_m_metric_x1, \
            current_t_metric_x2, \
            current_m_metric_x2 = model.t_metric_eval(num_samples=num_samples, context=context, transform_fct=transform_fct, cm_flow=cm_flow, device=args.device)
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
