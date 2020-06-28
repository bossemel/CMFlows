import torch
import torch.nn as nn
import scipy
from utils.various import sigmoid, t_m_metric_eval, flow_density, js_divergence
import datasets

eps = 0.0001


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

    def log_density(self, inputs):
        outputs, log_jacob = self(inputs=inputs)
        density = flow_density(outputs, log_jacob)
        return density

    def loss(self, inputs):
        return - self.log_density(inputs)

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
        # samples[samples >= 1] = 1 - eps
        # samples[samples <= 0] = 0 + eps
        return samples

    def jsd(self, args, inputs, transform_fct, obs=1000, cm_flow=False):
        # Define distributions
        normal_distr = scipy.stats.norm(0, 1)
        true_cop_distr = datasets.distributions.copula_distr(args.copula, args.theta)

        # Samples from both distributinos
        samples_pred = self.sample_copula(num_samples=inputs.shape[0], noise=None)
        if transform_fct == 'sigmoid':
            samples_target = torch.tensor(sigmoid(inputs))
        elif transform_fct == 'gaussian':
            samples_target = torch.tensor(normal_distr.cdf(inputs.cpu().numpy())).float()
        else:
            samples_target = torch.tensor(inputs)

        # Estimate Copula distr
        pred_distr = scipy.stats.gaussian_kde(samples_pred.T.cpu().numpy())

        # Prob X in both distributions
        prob_X_in_p = pred_distr.pdf(samples_pred.T.cpu().numpy()).T
        prob_X_in_q = true_cop_distr.pdf(samples_pred.cpu().numpy())

        # Prob Y in both distributions
        prob_Y_in_q = true_cop_distr.pdf(samples_target.numpy())
        prob_Y_in_p = pred_distr.pdf(samples_target.T).T

        assert samples_pred.cpu().numpy().all() > 0
        assert samples_target.cpu().numpy().all() > 0
        assert prob_X_in_p.all() > 0
        assert prob_X_in_q.all() > 0
        assert prob_Y_in_p.all() > 0
        assert prob_Y_in_q.all() > 0

        divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                   prob_X_in_q=prob_X_in_q,
                                   prob_Y_in_p=prob_Y_in_p,
                                   prob_Y_in_q=prob_Y_in_q)
        return divergence

    def t_metric_eval(self, num_samples, transform_fct, intervals=25, cm_flow=True):
        if cm_flow:
            samples = self.sample_copula(num_samples=num_samples, noise=None).detach().cpu().numpy()
        else:
            samples = self.sample(num_samples=num_samples, noise=None)
        samples = samples.cpu().numpy()
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
