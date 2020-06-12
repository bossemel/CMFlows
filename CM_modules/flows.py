import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import scipy.special
from utils.various import logit, sigmoid


class CMFlow(nn.Module):
    def __init__(self, transform, model_RealNVP, model_DDSF_1, model_DDSF_2):
        super(CMFlow, self).__init__()
        self.model_RealNVP = model_RealNVP
        self.model_DDSF_1 = model_DDSF_1
        self.model_DDSF_2 = model_DDSF_2
        self.transform_fct = transform

    def forward(self, xx, context=None):
        # @Todo: implement gaussian cdf transform
        # xx = logit(xx)

        # if self.transform_fct == 'gaussian':
        #     norm = scipy.stats.norm()
        #     xx = norm.ppf(xx)

        inputs, logdets = self.model_RealNVP(xx)

        # if self.transform_fct == 'sigmoid':
        inputs = sigmoid(inputs)
        # if self.transform_fct == 'gaussian':
        #     norm = scipy.stats.norm()
        #     inputs = norm.cdf(inputs)
        n = xx.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_())
        inputs_1, logdets_1, __ = self.model_DDSF_1((inputs[:, 0], logdets, context))
        inputs_2, logdets_2, __ = self.model_DDSF_2((inputs[:, 1], logdets, context))

        xx = torch.cat((inputs_1, inputs_2), dim=1)
        logdets = logdets_1 + logdets_2
        return xx, logdets

    def log_probs(self, inputs):
        n = inputs.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_())

        u, log_jacob = self(inputs, context)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        # if cond_inputs is not None:
        #     cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, mode='inverse')[0]
        return samples
