import datasets.distributions
import numpy as np
import torch


def jsd_eval(args, epoch, model, loader, device, test_dict,
             transform_model_1=None, transform_model_2=None, transform_inputs=True,
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

    if cm_flow is True:
        dataset = datasets.distributions.Copula_Distr(args, transform=False)
        data = dataset.tst
        with torch.no_grad():
            data = torch.tensor(data)
            current_jsd = model.jsd(args=args, inputs=data, transform_fct=args.transform_fct, cm_flow=cm_flow).sum().item()
        if 'jsd_test_copula' in test_dict:
            test_dict["jsd_test_copula"].append(current_jsd)
        else:
            test_dict["jsd_test_copula"] = [current_jsd]
    else:
        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            with torch.no_grad():
                current_jsd = model.jsd(args=args, inputs=data, transform_fct=args.transform_fct, cm_flow=cm_flow).sum().item()
            if 'jsd_test_copula' in test_dict:
                test_dict["jsd_test_copula"].append(current_jsd)
            else:
                test_dict["jsd_test_copula"] = [current_jsd]

    print('JSD in epoch {}:  {:5f}'.format(epoch, np.mean(test_dict["jsd_test_copula"])))
    return test_dict


def margin_uniformity(epoch, model, loader, device, transform_fct, test_dict, num_samples, cm_flow=False):
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
        current_t_metric_x1, \
            current_m_metric_x1, \
            current_t_metric_x2, \
            current_m_metric_x2 = model.t_metric_eval(num_samples=num_samples, transform_fct=transform_fct, cm_flow=cm_flow)
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
