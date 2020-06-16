import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from utils.various import sigmoid, logit


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

        Returns:
            inputs: inputs after forward pass
            logdets: sum of the log of the determinant of the jacobian of the models
            context: context parameter for DDSF
        """
        # @Todo: implement gaussian cdf transform
        inputs, logdets, context = inputs

        # The inputs are split and fed to each of the DDSF models
        inputs_1, logdets_1, __ = self.model_DDSF_1((inputs[:, 0].reshape(-1, 1), logdets, context))
        inputs_2, logdets_2, __ = self.model_DDSF_2((inputs[:, 1].reshape(-1, 1), logdets, context))

        # The outputs of the DDSF are concatenated to form a bivariate distribution
        xx = torch.cat((inputs_1, inputs_2), dim=1)

        # RealNVP is preceded by a logit and proceded by a sigmoid function
        if self.transform_fct == 'sigmoid':
            inputs = logit(inputs)
        if self.transform_fct == 'gaussian':
            raise NotImplementedError

        # forward pass in RealNVp
        inputs, logdets_RealNVP = self.model_RealNVP(xx)

        if self.transform_fct == 'sigmoid':
            inputs = sigmoid(inputs)
        if self.transform_fct == 'gaussian':
            raise NotImplementedError

        # log of the determinants of the jacobian are summed up
        logdets = logdets_1.reshape(-1, 1) + logdets_2.reshape(-1, 1) + logdets_RealNVP
        # print('realNVP', logdets_RealNVP.mean())
        # print('ddsf1', logdets_1.mean())
        # print('ddsf2', logdets_2.mean())
        return inputs, logdets, context

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
        u, log_jacob, __ = self((inputs, self.logdets, self.context))
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi))
        return log_probs + log_jacob

    def loss(self, x):
        """Loss is negative log density
        """
        loss = - self.log_density(x)
        assert loss.shape == x.shape
        return loss

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
