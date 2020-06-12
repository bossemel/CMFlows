import torch.nn as nn
import torch
import math


class CMFlow(nn.Module):
    def __init__(self, model_RealNVP, model_DDSF_1, model_DDSF_2):
        super(CMFlow, self).__init__()
        self.model_RealNVP = model_RealNVP
        self.model_DDSF_1 = model_DDSF_1
        self.model_DDSF_2 = model_DDSF_2

    def forward(self, xx):
        inputs, logdets = self.model_RealNVP(xx)
        inputs_1, logdets_1 = self.model_DDSF_1(inputs[:, 0])
        inputs_2, logdets_2 = self.model_DDSF_2(inputs[:, 1])

        xx = torch.cat((inputs_1, inputs_2), dim=1)
        logdets = logdets + logdets_1 + logdets_2
        return xx, logdets

    def log_probs(self, inputs):
        u, log_jacob = self(inputs)
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
