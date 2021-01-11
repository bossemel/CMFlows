import torch
import numpy as np
import math
import sys
import os
from sklearn import model_selection
from torch.autograd import Variable
import scipy.special
import scipy.stats
from utils.visualizer import visualize_joint
eps = 1e-3


def gaussian_change_of_var_ND(inputs, original_pdf, device, context=None):
    inputs[inputs == 0] = eps
    inputs[inputs == 1] = 1 - eps
    normal_distr = scipy.stats.norm()
    assert np.max(inputs) < 1, '{}'.format(np.max(inputs))
    assert np.min(inputs) > 0, '{}'.format(np.min(inputs))

    recast_inputs = torch.from_numpy(normal_distr.ppf(inputs)).float().to(device)

    if context is not None:
        context[context == 0] = eps
        context[context == 1] = 1 - eps
        assert np.max(context) < 1, '{}'.format(np.max(context))
        assert np.min(context) > 0, '{}'.format(np.min(context))
        recast_context = torch.from_numpy(normal_distr.ppf(context)).float().to(device)
        copy_recast_inputs = recast_inputs.detach().clone()
        original_joint = np.array(original_pdf(copy_recast_inputs, context=recast_context))
    else:
        copy_recast_inputs = recast_inputs.detach().clone()
        original_joint = np.array(original_pdf(copy_recast_inputs))

    if context is not None:
        recast_inputs = torch.cat([recast_inputs, recast_context], axis=1)
    second_dim = recast_inputs.shape[1] if len(recast_inputs.shape) ==2 else 1

    if second_dim >= 2:
        determinant = normal_distr.pdf(recast_inputs.cpu()).prod(axis=1) #.reshape(-1,)
    else:
        determinant = normal_distr.pdf(recast_inputs.cpu()) #.reshape(-1,)

    output = original_joint / determinant
    assert not np.isnan(output.sum())
    assert not np.isinf(output.sum())
    assert np.min(output) >= 0, '{}'.format(np.min(output))
    return output


def calc_jsd(args, test_dict, samples_pred, samples_target, name=''):
    visualize_joint(samples_target, args.figures_path, name='samples_target_jsd_{}'.format(name))
    visualize_joint(samples_pred, args.figures_path, name='samples_pred_jsd_{}'.format(name))

    # Define distributions
    pred_distr = scipy.stats.gaussian_kde(samples_pred.T)
    true_cop_distr = scipy.stats.gaussian_kde(samples_target.T)

    # Prob X in both distributions
    prob_X_in_p = pred_distr.pdf(samples_pred.T).T
    prob_X_in_q = true_cop_distr.pdf(samples_pred.T).T

    # Prob Y in both distributions
    prob_Y_in_q = true_cop_distr.pdf(samples_target.T).T
    prob_Y_in_p = pred_distr.pdf(samples_target.T).T

    assert not np.isnan(np.sum(prob_X_in_p))
    assert not np.isnan(np.sum(prob_X_in_q)), '%r' % (prob_X_in_q[:10])
    assert not np.isnan(np.sum(prob_Y_in_p))
    assert not np.isnan(np.sum(prob_Y_in_q)), '%r' % (prob_Y_in_q[:10])

    assert np.min(prob_X_in_p) >= 0
    assert np.min(prob_X_in_q) >= 0, '%r' % np.min(prob_X_in_q)
    assert np.min(prob_Y_in_p) >= 0
    assert np.min(prob_Y_in_q) >= 0

    divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                               prob_X_in_q=prob_X_in_q,
                               prob_Y_in_p=prob_Y_in_p,
                               prob_Y_in_q=prob_Y_in_q)
    print('JS-Divergence: {} {}'.format(divergence, name))
    test_dict['js_divergence'] = divergence
    return test_dict


def sigmoid(xx):
    """Sigmoid function in torch.
    """
    return 1 / (1 + torch.exp(-xx))


def logit(xx):
    """Logit function in torch.
    """
    return torch.log(xx / (1 - xx))


def t_m_metric_eval(margin, intervals=25):
    """Calculate T and M metric using Monte Carlo.
    Params:
        margin: margin array
        intervals: number of intervals to split data into (default 25)
    Returns:
        t_metric: int metric for marginal
        m_metric: int metric for marginal
    """
    sum_probs = 0
    highest_interval = 0
    for ii in range(intervals):
        A_k_lower = (ii - 1) / intervals
        A_k_upper = ii / intervals
        points_within = np.where(np.logical_and(margin >= A_k_lower, margin <= A_k_upper))[0]
        if len(points_within) > 0:
            prob = len(points_within) / len(margin)
            sum_probs += abs(np.log(prob) + np.log(intervals))
        else:
            prob = 0
        if prob > highest_interval:
            highest_interval = prob
    t_metric = sum_probs / intervals
    m_metric = abs(np.log(highest_interval) + np.log(intervals))
    return t_metric, m_metric


class HiddenPrints:
    """Hide Prints (for grid search).
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def flow_density(inputs, log_jacob):
    """Calculate density using change of variable formula.
    Params:
        inputs: data array
        log_jacob: accumulated log jacobian determinant
    Returns:
        log density array
    """
    log_prob = (-0.5 * inputs.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    return (log_prob + log_jacob).sum(-1, keepdim=True)


def js_divergence_grid(prob_vector_X, prob_vector_Y):
    """Calculates the JS-Divergence of each point on a grid using pointwise KL-divergence, averaged.
    Params:
        prob_vector_X: probability of grid in distr p(x)
        prob_vector_Y: probability of grid in distr p(y)
    Returns:
        pointwise JS-Divergence
    """
    mix = 0.5 * (prob_vector_Y + prob_vector_X)
    KL_X_mix = scipy.special.rel_entr(prob_vector_X, mix).mean()
    KL_Y_mix = scipy.special.rel_entr(prob_vector_Y, mix).mean()
    return (KL_X_mix + KL_Y_mix) / 2


def js_divergence(prob_X_in_p, prob_X_in_q,
                  prob_Y_in_p, prob_Y_in_q):
    """Calculate JS-Divergence using Monte Carlo.
    Params:
        prob_X_in_p: p(x), x from distr p(x), array
        prob_X_in_q: q(x), x from distr p(x), array
        prob_Y_in_p: p(y), y from distr q(y), array
        prob_Y_in_q: p(y), y from distr q(y), array
    Returns:
        divergence: int, JS-Divergence
    """

    mix_X = prob_X_in_p + prob_X_in_q
    mix_Y = prob_Y_in_p + prob_Y_in_q

    #mix_X[mix_X == 0] = 0 + eps
    #mix_Y[mix_Y == 0] = 0 + eps

    prob_X_in_p[prob_X_in_p == 0] = 0 + eps
    prob_Y_in_q[prob_Y_in_q == 0] = 0 + eps

    assert np.min(mix_X) > 0
    assert np.min(mix_Y) > 0

    KL_PM = np.log2((2 * prob_X_in_p) / mix_X)

    KL_PM[mix_X == 0] = 0
    KL_PM = KL_PM.mean()

    KL_QM = np.log2((2 * prob_Y_in_q) / mix_Y)

    KL_QM[mix_Y == 0] = 0
    KL_QM = KL_QM.mean()

    divergence = (KL_PM + KL_QM) / 2
    assert divergence >= 0

    return divergence


def split_train_val_test(xx, only_val=False):
    """Splits data into train, val and test set, using 80/20/280 split.
    Params:
        xx: data to split
    Returns:
        train, val, test: train, val and test set
    """
    if only_val:
        train, val = model_selection.train_test_split(xx, test_size=0.2)
        return train, val
    else:
        train, testval = model_selection.train_test_split(xx, test_size=0.2)
        val, test = model_selection.train_test_split(testval, test_size=0.5)
        return train, val, test


def empty_logdets_context(inputs, device):
    """Create empty arrays as inputs for DDSF.
    """
    n = inputs.shape[0]
    context = Variable(torch.FloatTensor(n, 1).zero_()).to(device)
    logdets = Variable(torch.FloatTensor(n).zero_()).to(device)
    return logdets, context


def normalize(dataset):
    mean, std = np.mean(dataset), np.std(dataset)
    dataset = dataset - mean
    dataset = dataset / std
    return dataset

def normalize_torch(dataset):
    mean, std = torch.mean(dataset), torch.std(dataset)
    dataset = dataset - mean
    dataset = dataset / std
    return dataset


def make_meshgrid(obs, dim, low, high):
    meshgrid = np.array(np.meshgrid(*[np.linspace(low, high, obs)] * dim))
    return np.concatenate([vector.reshape(-1, 1) for vector in meshgrid], axis=1)
