import torch
import torch.nn as nn
import scipy
from utils import sigmoid, t_m_metric_eval, flow_density, js_divergence
import numpy as np
from utils.visualizer import visualize_joint


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)
        if self.num_inputs == 1:
            inputs = inputs.reshape(-1, 1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs=inputs, cond_inputs=cond_inputs, mode=mode)
                logdets += logdet

        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs=inputs, cond_inputs=cond_inputs, mode=mode)
                logdets += logdet

        return inputs, logdets

    def log_density(self, inputs, cond_inputs=None):
        """Calculates log density of the flow
        """
        outputs, log_jacob = self(inputs=inputs, cond_inputs=cond_inputs)
        density = flow_density(outputs, log_jacob)
        return density

    def loss(self, inputs, cond_inputs=None):
        """Return negative log likelihood/density
        """
        return - self.log_density(inputs, cond_inputs)

    def transform(self, inputs, cond_inputs, mode='direct', device=None):
        if device is not None:
            inputs = inputs.to(device)
            cond_inputs = cond_inputs.to(device)
        return self.forward(inputs=inputs, cond_inputs=cond_inputs, mode=mode)[0]

    def sample(self, num_samples=None, transform=None, cond_inputs=None, num_inputs=None, copula=False, device=None):
        """Returns an output sample without transformation
        """
        if num_inputs is not None:
            self.num_inputs = num_inputs
        if cond_inputs is not None:
            num_samples = cond_inputs.shape[0]
        noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        if device is not None:
            if cond_inputs is not None:
                cond_inputs = cond_inputs.to(device)
            noise = noise.to(device)
        samples = self.forward(inputs=noise, cond_inputs=cond_inputs, mode='inverse')[0]
        if cond_inputs is not None:
            samples = torch.cat([cond_inputs, samples], axis=1)
        if not copula:
            if transform == 'sigmoid':
                samples = sigmoid(samples)
            elif transform == 'gaussian':
                normal_distr = torch.distributions.normal.Normal(0, 1)
                samples = normal_distr.cdf(samples)
        else:
            normal_distr = torch.distributions.normal.Normal(0, 1)
            samples = normal_distr.cdf(samples)
        return samples

    def sample_copula(self, num_samples=None, cond_inputs=None, num_inputs=None, device=None):
        """Returns the predicted copula (output sample with transformation)
        """
        if num_inputs is not None:
            self.num_inputs = num_inputs
        noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        if device is not None:
            noise = noise.to(device)
            if cond_inputs is not None:
                cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs=cond_inputs, mode='inverse')[0]
        if cond_inputs is not None:
            samples = torch.cat([cond_inputs, samples], axis=1)
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        return samples

    def jsd(self, args, inputs, cond_inputs=None, transform_fct=None, obs=1000, cm_flow=False):
        """Returns JS-Divergence of the predicted Copula and the true Copula
        """
        with torch.no_grad():
            samples_target = torch.tensor(inputs)
            # Define distributions
            normal_distr = scipy.stats.norm(0, 1)
            # true_cop_distr = datasets.distributions.Copula_Distr(args=args, transform=False)

            # Samples from both distributinos
            print(inputs.shape)
            if cm_flow is True:
                samples_pred = self.sample_copula(num_samples=inputs.shape[0], cond_inputs=cond_inputs, device=args.device)
                samples_pred_viz = self.sample_copula(num_samples=10000, cond_inputs=cond_inputs, device=args.device)
            else:
                samples_pred = self.sample(num_samples=inputs.shape[0], cond_inputs=cond_inputs, transform=transform_fct, device=args.device)
                samples_pred_viz = self.sample(num_samples=10000, cond_inputs=cond_inputs, transform=transform_fct, device=args.device)

            if not cm_flow:
                if transform_fct == 'sigmoid':
                    samples_target = sigmoid(inputs)
                    if args.conditional_copula:
                        cond_inputs = sigmoid(cond_inputs)
                elif transform_fct == 'gaussian':
                    normal_distr = torch.distributions.normal.Normal(0, 1)
                    samples_target = normal_distr.cdf(inputs)
                    if args.conditional_copula:
                        cond_inputs = normal_distr.cdf(cond_inputs)
            else:
                normal_distr = torch.distributions.normal.Normal(0, 1)
                samples_target = normal_distr.cdf(inputs)
                if args.conditional_copula:
                    cond_inputs = normal_distr.cdf(cond_inputs)

            if args.conditional_copula:
                visualize_joint(torch.cat([samples_target, cond_inputs], axis=1).cpu(), args, name='samples_target_jsd')
                visualize_joint(torch.cat([samples_pred_viz, cond_inputs], axis=1).cpu(), args, name='samples_pred_jsd')
            else:
                visualize_joint(samples_target.cpu(), args, name='samples_target_jsd')
                visualize_joint(samples_pred_viz.cpu(), args, name='samples_pred_jsd')

            assert np.max(samples_target.cpu().numpy()) <= 1
            assert np.min(samples_target.cpu().numpy()) >= 0
            assert np.max(samples_pred.cpu().numpy()) <= 1
            assert np.min(samples_pred.cpu().numpy()) >= 0
            # Prob X in both distributions
            pred_distr = scipy.stats.gaussian_kde(samples_pred.T.cpu())

            if args.conditional_copula:
                true_cop_distr = scipy.stats.gaussian_kde(torch.cat([samples_target, cond_inputs], axis=1).cpu().numpy().T)
            else:
                true_cop_distr = scipy.stats.gaussian_kde(samples_target.T.cpu())

            prob_X_in_p = pred_distr.pdf(samples_pred.cpu().numpy().T).T

            if args.conditional_copula:
                prob_X_in_q = true_cop_distr.pdf(samples_pred.T.cpu()).T
            else:
                prob_X_in_q = true_cop_distr.pdf(samples_pred.cpu().numpy().T).T

            # Prob Y in both distributions
            if args.conditional_copula:
                prob_Y_in_q = true_cop_distr.pdf(np.concatenate([samples_target.cpu().numpy(), cond_inputs.cpu()], axis=1).T).T
                prob_Y_in_p = pred_distr.pdf(torch.cat([samples_target.cpu(), cond_inputs.cpu()], axis=1).cpu().numpy().T).T
            else:
                prob_Y_in_q = true_cop_distr.pdf(samples_target.cpu().numpy().T).T
                prob_Y_in_p = pred_distr.pdf(samples_target.cpu().numpy().T).T

            assert np.min(prob_X_in_p) >= 0
            assert np.min(prob_X_in_q) >= 0
            assert np.min(prob_Y_in_p) >= 0
            assert np.min(prob_Y_in_q) >= 0

            divergence = js_divergence(prob_X_in_p=prob_X_in_p.reshape(-1,),
                                       prob_X_in_q=prob_X_in_q.reshape(-1,),
                                       prob_Y_in_p=prob_Y_in_p.reshape(-1,),
                                       prob_Y_in_q=prob_Y_in_q.reshape(-1,))

            return divergence

    def t_metric_eval(self, args, num_samples, cond_inputs=None, transform_fct=None, intervals=25, cm_flow=False):
        """Returns evaluation metrics for the copula marginals.
        """
        with torch.no_grad():
            if cm_flow:
                if args.conditional_copula:
                    samples = self.sample_copula(num_samples=num_samples, cond_inputs=cond_inputs, device=args.device).cpu().numpy()
                else:
                    samples = self.sample_copula(num_samples=num_samples, device=args.device).cpu().numpy()
            else:
                if args.conditional_copula:
                    samples = self.sample(num_samples=num_samples, cond_inputs=cond_inputs, transform=transform_fct, device=args.device).cpu().numpy()
                else:
                    samples = self.sample(num_samples=num_samples, transform=transform_fct, device=args.device).cpu().numpy()
            if args.conditional_copula:
                margin_x1 = cond_inputs
                margin_x2 = samples
            else:
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
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        assert True, 'coupling layer initialized'
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
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

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

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

    def forward(self, inputs, cond_inputs=None, mode='direct'):
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
