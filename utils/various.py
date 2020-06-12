import torch


def sigmoid(xx):
    return 1 / (1 + torch.exp(-xx))


def logit(xx):
    return torch.log(xx / (1 - xx))
