import torch
import numpy as np


def sigmoid(xx):
    return 1 / (1 + torch.exp(-xx))


def logit(xx):
    return torch.log(xx / (1 - xx))


def t_m_metric_eval(margin, intervals):
    sum_probs = 0
    highest_interval = 0
    for ii in range(intervals):
        A_k_lower = (ii - 1) / intervals
        A_k_upper = ii / intervals
        points_within = np.where(np.logical_and(margin >= A_k_lower, margin <= A_k_upper))[0]
        if len(points_within) > 0:
            log_prob = np.log(points_within.sum() / len(points_within))
        else:
            log_prob = 0
        if log_prob > highest_interval:
            highest_interval = log_prob
        sum_probs += abs(log_prob + np.log(intervals))
        t_metric = sum_probs / intervals
        m_metric = (highest_interval + np.log(intervals)) / intervals
    return t_metric, m_metric
