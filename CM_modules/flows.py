import torch.nn as nn
import torch
import scipy
from utils.various import t_m_metric_eval, js_divergence
import datasets
import numpy as np
eps = 0.0001


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
        self.cuda = args.cuda
        self.copula = args.copula
        self.theta = args.theta

    def forward_DDSF_1(self, inputs):
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
        inputs_, logdets, context = inputs

        # The inputs are split and fed to each of the DDSF models
        outputs_DDSF_1, logdets_DDSF_1, __ = self.model_DDSF_1((inputs_[:, 0].reshape(-1, 1), logdets, context))
        return outputs_DDSF_1, logdets_DDSF_1

    def forward_DDSF_2(self, inputs):
        inputs_, logdets, context = inputs
        outputs_DDSF_2, logdets_DDSF_2, __ = self.model_DDSF_2((inputs_[:, 1].reshape(-1, 1), logdets, context))
        return outputs_DDSF_2, logdets_DDSF_2

    def forward_RealNVP(self, inputs, logdets=None, mode='direct'):
        # forward pass in RealNVP
        outputs_RealNVP, logdets_RealNVP = self.model_RealNVP(inputs=inputs, logdets=logdets, mode=mode)
        return outputs_RealNVP, logdets_RealNVP

    def sample(self, num_samples=None, noise=None):
        """Samples from the copula without transforming the marginals.
        """
        if noise is None:
            noise = torch.Tensor(num_samples, 1).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward_RealNVP(noise, mode='inverse')[0]
        return samples

    def sample_copula(self, num_samples=None, noise=None):
        """Sampels from the copula and transforms the marginals to uniform
        """
        if noise is None:
            noise = torch.Tensor(num_samples, 1).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward_RealNVP(noise, mode='inverse')[0]
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        return samples

    def jsd(self, args, inputs, transform_fct, obs=1000, cm_flow=False):
        """Evaluated the JS-Divergence using Monte Carlo.
        """
        # Samples from both distributinos
        samples_pred = self.sample_copula(num_samples=inputs.shape[0], noise=None)
        samples_pred = samples_pred.detach().cpu().numpy()

        assert np.min(samples_pred) >= 0
        assert np.max(samples_pred) <= 1

        samples_target = inputs
        samples_target[samples_target == 1] = 1 - eps
        samples_target[samples_target == 0] = 0 + eps
        samples_pred[samples_pred == 0] = 0 + eps
        samples_pred[samples_pred == 1] = 1 - eps

        assert np.min(samples_target) > 0
        assert np.max(samples_target) < 1
        assert np.min(samples_pred) > 0
        assert np.max(samples_pred) < 1, '%r' % (np.max(samples_pred))

        samples_target = torch.tensor(inputs)

        # Define distributions
        true_cop_distr = datasets.distributions.Copula_Distr(args=args, transform=False)
        # Estimate Copula distr
        pred_distr = scipy.stats.gaussian_kde(samples_pred.T)

        # Prob X in both distributions
        prob_X_in_p = pred_distr.pdf(samples_pred.T).T
        prob_X_in_q = true_cop_distr.pdf(samples_pred)

        # Prob Y in both distributions
        prob_Y_in_q = true_cop_distr.pdf(samples_target.detach().cpu().numpy())
        prob_Y_in_p = pred_distr.pdf(samples_target.T).T

        if np.isnan(np.sum(prob_X_in_q)):
            prob_X_in_p = prob_X_in_p[~np.isnan(prob_X_in_q)]
            prob_Y_in_q = prob_Y_in_q[~np.isnan(prob_X_in_q)]
            prob_Y_in_p = prob_Y_in_p[~np.isnan(prob_X_in_q)]
            prob_X_in_q = prob_X_in_q[~np.isnan(prob_X_in_q)]

        if np.isnan(np.sum(prob_Y_in_q)):
            prob_X_in_p = prob_X_in_p[~np.isnan(prob_Y_in_q)]
            prob_X_in_q = prob_X_in_q[~np.isnan(prob_Y_in_q)]
            prob_Y_in_p = prob_Y_in_p[~np.isnan(prob_Y_in_q)]
            prob_Y_in_q = prob_Y_in_q[~np.isnan(prob_Y_in_q)]

        prob_Y_in_p[prob_Y_in_p == 0] = 0 + eps
        prob_X_in_p[prob_X_in_q == 0] = 0 + eps
        prob_X_in_p[prob_Y_in_p == 0] = 0 + eps
        prob_X_in_p[prob_Y_in_q == 0] = 0 + eps

        assert not np.isnan(np.sum(prob_X_in_p))
        assert not np.isnan(np.sum(prob_X_in_q)), '%r' % (prob_X_in_q[:10])
        assert not np.isnan(np.sum(prob_Y_in_p))
        assert not np.isnan(np.sum(prob_Y_in_q)), '%r' % (prob_Y_in_q[:10])

        assert np.min(prob_X_in_p) > 0
        assert np.min(prob_X_in_q) > 0, '%r' % np.min(prob_X_in_q)
        assert np.min(prob_Y_in_p) > 0
        assert np.min(prob_Y_in_q) > 0

        divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                   prob_X_in_q=prob_X_in_q,
                                   prob_Y_in_p=prob_Y_in_p,
                                   prob_Y_in_q=prob_Y_in_q)
        return divergence

    def t_metric_eval(self, num_samples, transform_fct, intervals=25, cm_flow=False):
        """Evaluates the uniformity of the predicted marginals.
        """
        if cm_flow:
            samples = self.sample_copula(num_samples=num_samples, noise=None).detach().cpu().numpy()
        margin_x1 = samples[:, 0]
        margin_x2 = samples[:, 1]
        t_metric_x1, m_metric_x1 = t_m_metric_eval(margin_x1, intervals)
        t_metric_x2, m_metric_x2 = t_m_metric_eval(margin_x2, intervals)
        return t_metric_x1, m_metric_x1, t_metric_x2, m_metric_x2

    def clip_grad_norm(self):
        """Performs gradient clipping
        """
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)
