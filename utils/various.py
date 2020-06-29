import torch
import numpy as np
import math
import sys
import os
from sklearn import model_selection


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
            log_prob = np.log(len(points_within) / len(margin))
            assert not math.isinf(log_prob), (len(points_within) / len(margin))
        else:
            log_prob = 0
        if log_prob > highest_interval:
            highest_interval = log_prob
        sum_probs += abs(log_prob + np.log(intervals))
        t_metric = sum_probs / intervals
        m_metric = (highest_interval + np.log(intervals))
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
    log_prob = (-0.5 * inputs.pow(2) - 0.5 * math.log(2 * math.pi))
    return (log_prob + log_jacob).sum(-1, keepdim=True)


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
    mix_X = np.logaddexp(prob_X_in_p, prob_X_in_q)
    mix_Y = np.logaddexp(prob_Y_in_p, prob_Y_in_q)

    KL_PM = np.log2(2) + np.log2(prob_X_in_p).mean() - np.log2(mix_X).mean()
    KL_QM = np.log2(2) + np.log2(prob_Y_in_q).mean() - np.log2(mix_Y).mean()

    divergence = (KL_PM + KL_QM) / 2
    return divergence


def split_train_val_test(xx):
    """Splits data into train, val and test set, using 80/20/280 split.

    Params:
        xx: data to split

    Returns:
        train, val, test: train, val and test set
    """
    train, testval = model_selection.train_test_split(xx, test_size=0.2)
    val, test = model_selection.train_test_split(testval, test_size=0.5)
    return train, val, test
