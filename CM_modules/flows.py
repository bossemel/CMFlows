import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from utils.various import sigmoid, logit
import scipy
import matplotlib.pyplot as plt


def flow_density(inputs, log_jacob):
    log_prob = (-0.5 * inputs.pow(2) - 0.5 * math.log(2 * math.pi))
    return log_prob + log_jacob


class CMFlow(nn.Module):
    def __init__(self, transform, model_RealNVP, model_DDSF_1, model_DDSF_2,
                 device, batch_size, args):
        super(CMFlow, self).__init__()
        self.model_RealNVP = model_RealNVP
        self.model_DDSF_1 = model_DDSF_1
        self.model_DDSF_2 = model_DDSF_2
        self.transform_fct = transform
        self.device = device
        self.batch_size = batch_size
        self.args = args
        self.clip = self.args.clip

    def forward(self, inputs):
        """Forward pass of CM Flows model. The inputs are first passed
        through the two DDSF models, which project onto a uniform distributions.
        The RealNVP then uses the CM Flows output to map onto a normal distribution.

        Params:
            inputs: joint distribution samples

        Returns:z
            inputs: inputs after forward pass
            logdets: sum of the log of the determinant of the jacobian of the models
            context: context parameter for DDSF
        """
        # @Todo: implement gaussian cdf transform
        eps = 0.00001
        inputs, logdets, context = inputs
        # inputs_clone = inputs.clone()
        # logdets_clone = logdets.clone()
        # context_clone = context.clone()

        # The inputs are split and fed to each of the DDSF models
        outputs_DDSF_1, logdets_DDSF_1, __ = self.model_DDSF_1((inputs[:, 0].reshape(-1, 1), logdets, context))
        outputs_DDSF_2, logdets_DDSF_2, __ = self.model_DDSF_2((inputs[:, 1].reshape(-1, 1), logdets, context))

        normal_distr = torch.distributions.normal.Normal(0, 1)
        outputs_DDSF_1 = normal_distr.cdf(outputs_DDSF_1)
        outputs_DDSF_2 = normal_distr.cdf(outputs_DDSF_2)

        # The outputs of the DDSF are concatenated to form a bivariate distribution
        outputs_DDSFs = torch.cat((outputs_DDSF_1, outputs_DDSF_2), dim=1)
        outputs_DDSFs[outputs_DDSFs >= 1] = 1 - eps
        outputs_DDSFs[outputs_DDSFs <= 0] = 0 + eps
        # outputs_DDSFs_clone = outputs_DDSFs.clone()

        if self.transform_fct == 'sigmoid':
            outputs_DDSFs = logit(outputs_DDSFs)
        if self.transform_fct == 'gaussian':
            raise NotImplementedError

        # forward pass in RealNVp
        outputs_RealNVP, logdets_RealNVP = self.model_RealNVP(outputs_DDSFs)

        logdets = (logdets_DDSF_1.reshape(-1, 1), logdets_DDSF_2.reshape(-1, 1), logdets_RealNVP)
        outputs = (outputs_DDSF_1, outputs_DDSF_2, outputs_RealNVP)
        return outputs, logdets

    def log_density(self, inputs):
        """Returns log of target density of the Flow

        Params:
            inputs: target distribution samples

        Returns:
            log density of the model
        """
        self.n = inputs.shape[0]
        self.context = Variable(torch.FloatTensor(self.n, 1).zero_()).to(self.device)
        self.logdets = Variable(torch.FloatTensor(self.n).zero_()).to(self.device)
        outputs, log_jacob = self((inputs, self.logdets, self.context))
        logdets_DDSF_1, logdets_DDSF_2, logdets_RealNVP = log_jacob
        outputs_DDSF_1, outputs_DDSF_2, outputs_RealNVP = outputs
        density_DDSF_1 = flow_density(outputs_DDSF_1, logdets_DDSF_1)
        density_DDSF_2 = flow_density(outputs_DDSF_2, logdets_DDSF_2)
        density_RealNVP = flow_density(outputs_RealNVP, logdets_RealNVP)

        return (density_DDSF_1, density_DDSF_2, density_RealNVP)

    def loss(self, x):
        """Loss is negative log density
        """
        density_DDSF_1, density_DDSF_2, density_RealNVP = self.log_density(x)
        return (-density_DDSF_1, -density_DDSF_2, -density_RealNVP)

    def sample(self, num_samples=None, noise=None):
        """Samples from the Distribution
        """
        raise NotImplementedError
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        # if cond_inputs is not None:
        #     cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, mode='inverse')[0]
        return samples

    def clip_grad_norm(self):
        """Performs gradient clipping
        """
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)
