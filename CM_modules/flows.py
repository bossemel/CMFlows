import torch.nn as nn
import torch
from utils import t_m_metric_eval, js_divergence, flow_density
import numpy as np
from RealNVP import build_model as build_model_RealNVP
from DDSF import build_model as build_model_DDSF
from NSF import build_model as build_model_nsf
import scipy.stats
from utils.visualizer import visualize_joint
import datasets.distribution
eps = 0.0001


class CMFlow(nn.Module):
    def __init__(self, transform,
                 device, batch_size, args):
        super(CMFlow, self).__init__()
        self.cop_flow = build_model_nsf(args, flow_type='cop_flow')
        self.marg_flow_1 = build_model_nsf(args, flow_type='marg_flow')
        self.marg_flow_2 = build_model_nsf(args, flow_type='marg_flow')
        # self.cop_flow = build_model_RealNVP(args)
        # self.marg_flow_1 = build_model_DDSF(args)
        # self.marg_flow_2 = build_model_DDSF(args)
        self.transform_fct = transform
        self.device = device
        self.batch_size = batch_size
        self.args = args
        self.clip = self.args.clip
        self.cuda = args.cuda
        self.copula = args.copula
        self.theta = args.theta

    def log_density_DDSF_1(self, inputs, logdets=None, context=None):
        """Returns log of target density of the Flow

        Params:
            inputs: target distribution samples

        Returns:
            log density of the model
        """
        return self.marg_flow_1._forward(inputs)

    def log_density_DDSF_2(self, inputs, logdets=None, context=None):
        """Returns log of target density of the Flow

        Params:
            inputs: target distribution samples

        Returns:
            log density of the model
        """
        return self.marg_flow_2._forward(inputs)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        """Samples from the copula without transforming the marginals.
        """
        if noise is None:
            noise = torch.Tensor(num_samples, 2).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.cop_flow.forward(inputs=noise, cond_inputs=cond_inputs, mode='inverse')[0]
        return samples

    def sample_copula(self, num_samples=None, noise=None):
        """Sampels from the copula and transforms the marginals to uniform
        """
        if noise is None:
            noise = torch.Tensor(num_samples, 2).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.cop_flow.forward(noise, mode='inverse')[0]
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        return samples

    def jsd(self, args, inputs, cond_inputs=None, transform_fct='gaussian', obs=1000, cm_flow=False):
        """Evaluated the JS-Divergence using Monte Carlo.
        """
        with torch.no_grad():
            # samples_target = inputs.cpu().numpy()
            # Samples from both distributinos
            normal_distr = torch.distributions.normal.Normal(0, 1)
            samples_target = normal_distr.cdf(inputs)

            if args.conditional_copula:
                cond_inputs = torch.tensor(normal_distr.cdf(cond_inputs))

            assert torch.min(samples_target) > 0
            assert torch.max(samples_target) < 1

            samples_pred = self.sample_copula(num_samples=samples_target.shape[0]).cpu().numpy()
            pred_distr = scipy.stats.gaussian_kde(samples_pred.T)

            # Define distributions
            # if args.conditional_copula:
            #     true_cop_distr = scipy.stats.gaussian_kde(torch.cat([samples_target, cond_inputs], axis=1).cpu().numpy().T)
            # else:
            #     true_cop_distr = scipy.stats.gaussian_kde(samples_target.T)
            # Get true copula distribution
            true_cop_distr = datasets.distributions.Copula_Distr(args)

            # Prob X in both distributions
            prob_X_in_p = pred_distr.pdf(samples_pred.T).T
            if args.conditional_copula:
                prob_X_in_q = true_cop_distr.pdf(samples_pred)
            else:
                prob_X_in_q = true_cop_distr.pdf(samples_pred)

            # Prob Y in both distributions
            if args.conditional_copula:
                prob_Y_in_q = true_cop_distr.pdf(np.concatenate([samples_target.cpu().numpy(), cond_inputs], axis=1))
                prob_Y_in_p = pred_distr.pdf(torch.cat([samples_target.cpu(), cond_inputs], axis=1).cpu().numpy().T).T

            else:
                prob_Y_in_q = true_cop_distr.pdf(samples_target.cpu().numpy())
                prob_Y_in_p = pred_distr.pdf(samples_target.cpu().numpy().T).T

            assert not np.isnan(np.sum(prob_X_in_p))
            assert not np.isnan(np.sum(prob_X_in_q)), '%r' % (prob_X_in_q[:10])
            assert not np.isnan(np.sum(prob_Y_in_p))
            assert not np.isnan(np.sum(prob_Y_in_q)), '%r' % (prob_Y_in_q[:10])

            assert np.min(prob_X_in_p) >= 0
            assert np.min(prob_X_in_q) >= 0, '%r' % np.min(prob_X_in_q)
            assert np.min(prob_Y_in_p) >= 0
            assert np.min(prob_Y_in_q) >= 0

            divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                       prob_X_in_q=prob_X_in_q,
                                       prob_Y_in_p=prob_Y_in_p,
                                       prob_Y_in_q=prob_Y_in_q)
        return divergence

    def t_metric_eval(self, args, num_samples, cond_inputs=None, transform_fct=None, intervals=25, cm_flow=None):
        """Evaluates the uniformity of the predicted marginals.
        """
        with torch.no_grad():
            samples = self.sample_copula(num_samples=num_samples).cpu().numpy()
            margin_x1 = samples[:, 0]
            margin_x2 = samples[:, 1]
            t_metric_x1, m_metric_x1 = t_m_metric_eval(margin_x1, intervals)
            t_metric_x2, m_metric_x2 = t_m_metric_eval(margin_x2, intervals)
            return t_metric_x1, m_metric_x1, t_metric_x2, m_metric_x2

    def clip_grad_norm(self):
        """Performs gradient clipping
        """
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)
