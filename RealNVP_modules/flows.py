import math
import torch
import torch.nn as nn
import numpy as np
import scipy
from utils.various import sigmoid, t_m_metric_eval

eps = 0.0001


def flow_density(inputs, log_jacob):
    log_prob = (-0.5 * inputs.pow(2) - 0.5 * math.log(2 * math.pi))
    return log_prob + log_jacob


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        if isinstance(inputs, tuple):
            inputs, __, __ = inputs
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs):
        outputs, log_jacob = self(inputs)
        density = flow_density(outputs, log_jacob)
        return density

        # u, log_jacob = self(inputs)
        # log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi))
        # return (log_probs + log_jacob)

    def loss(self, inputs):
        return - self.log_probs(inputs)

    def sample(self, num_samples=None, noise=None, transform=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward(noise, mode='inverse')[0]
        if transform == 'sigmoid':
            samples = sigmoid(samples)
        elif transform == 'gaussian':
            normal_distr = torch.distributions.normal.Normal(0, 1)
            samples = normal_distr.cdf(samples)
        return samples

    def sample_copula(self, num_samples=None, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward(noise, mode='inverse')[0]
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        samples[samples > 1] = 1 - eps
        samples[samples < 0] = 0 + eps
        return samples

    def jsd(self, inputs, transform_fct, obs=100, cm_flow=False):
        x1 = np.linspace(0, 1, obs)
        x2 = np.linspace(0, 1, obs)
        grid1, grid2 = np.meshgrid(x1, x2)
        grid1 = grid1.reshape(x1.shape[0] * x2.shape[0], 1)
        grid2 = grid2.reshape(x1.shape[0] * x2.shape[0], 1)
        grid2d = np.concatenate([grid1, grid2], axis=1)
        if cm_flow:
            samples = self.sample_copula(num_samples=obs**2, noise=None)
        else:
            samples = self.sample(num_samples=obs**2, noise=None, transform=transform_fct)
        if transform_fct == 'sigmoid':
            inputs = sigmoid(inputs)
        elif transform_fct == 'gaussian':
            normal_distr = torch.distributions.normal.Normal(0, 1)
            inputs = normal_distr.cdf(inputs)
        # print('samples', samples[:10])
        # print('inputs', inputs[:10])
        # print('samples.T', samples.T.shape)
        pred_pdf = scipy.stats.gaussian_kde(samples.T)
        pred_grid = pred_pdf(grid2d.T) # .reshape(obs, obs)
        true_pdf = scipy.stats.gaussian_kde(inputs.T)
        true_grid = true_pdf(grid2d.T) # .reshape(obs, obs)
        # print(pred_grid[:10], true_grid[:10])
        # print(pred_grid.shape, true_grid.shape)
        # plt.plot(pred_grid[:, 0], grid1)
        # plt.plot(true_grid[:, 0], grid1)
        # plt.show()
        assert np.min(pred_grid) >= 0
        assert np.min(true_grid) >= 0
        # assert np.max(pred_grid) <= 1, print(pred_grid[pred_grid>1])
        # assert np.max(true_grid) <= 1
        divergence = scipy.spatial.distance.jensenshannon(pred_grid, true_grid)
        # print('1d method: ', divergence)
        # divergence = js_divergence(pred_grid, true_grid)
        # print('my method', divergence)
        return divergence

    def t_metric_eval(self, num_samples, transform_fct, intervals=25, cm_flow=False):
        if cm_flow:
            samples = self.sample_copula(num_samples=num_samples, noise=None)
        else:
            samples = self.sample(num_samples=num_samples, noise=None)
            if transform_fct == 'sigmoid':
                samples = scipy.special.expit(samples.detach().cpu())
            if transform_fct == 'gaussian':
                norm = scipy.stats.norm()
                samples = norm.cdf(samples.cpu())
        margin_x1 = samples[:, 0]
        margin_x2 = samples[:, 1]
        t_metric_x1, m_metric_x1 = t_m_metric_eval(margin_x1, intervals)
        t_metric_x2, m_metric_x2 = t_m_metric_eval(margin_x2, intervals)
        return t_metric_x1, m_metric_x1, t_metric_x2, m_metric_x2


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        # if num_cond_inputs is not None:
        #     total_inputs = num_inputs + num_cond_inputs
        # else:
        total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, mode='direct'):
        # inputs = torch.log(inputs / (1 - inputs))
        # inputs = scipy.special.logit(inputs)
        mask = self.mask

        masked_inputs = inputs * mask
        # if cond_inputs is not None:
        #     masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
