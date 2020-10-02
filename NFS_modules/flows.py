import torch
import torch.nn as nn
import NFS_modules.nn as nn_
import NFS_modules.utils as utils
from NFS_modules.nde import distributions, flows, transforms
import scipy.stats
from utils.visualizer import visualize_joint
import numpy as np
from utils import js_divergence, t_m_metric_eval


class ConditionalFlow(nn.Module):
    """A conditional rational quadratic neural spline flow."""
    def __init__(self, dim, context_dim, n_layers, hidden_units, n_blocks, dropout,
                 use_batch_norm, tails, tail_bound, n_bins, min_bin_height,
                 min_bin_width, min_derivative, unconditional_transform, subsample, device):
        super().__init__()
        self.dim = dim
        self.num_inputs = dim
        self.context_dim = context_dim
        self.n_layers = n_layers
        # self.n_encoder_layers = n_encoder_layers
        # self.encoder_units = encoder_units
        self.hidden_units = hidden_units
        self.n_blocks = n_blocks
        self.dropout = dropout
        # self.encoder_dropout = encoder_dropout
        self.use_batch_norm = use_batch_norm
        self.tails = tails
        self.tail_bound = tail_bound
        self.n_bins = n_bins
        self.min_bin_height = min_bin_height
        self.min_bin_width = min_bin_width
        self.min_derivative = min_derivative
        self.unconditional_transform = unconditional_transform
        # self.use_cnn_encoder = use_cnn_encoder
        self.subsample = subsample
        self.device = device

        distribution = distributions.StandardNormal([dim]).to(device)
        transform = transforms.CompositeTransform([
            self.create_transform() for _ in range(self.n_layers)], device)
        self.flow = flows.Flow(transform, distribution).to(device)

    def create_transform(self):
        """Create invertible rational quadratic transformations."""
        linear = transforms.RandomPermutation(features=self.dim).to(self.device)
        base = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_mid_split_binary_mask(features=self.dim),
            transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    context_features=self.context_dim,
                    hidden_features=self.hidden_units,
                    num_blocks=self.n_blocks,
                    dropout_probability=self.dropout,
                    use_batch_norm=self.use_batch_norm,),
            tails=self.tails,
            tail_bound=self.tail_bound,
            num_bins=self.n_bins,
            min_bin_height=self.min_bin_height,
            min_bin_width=self.min_bin_width,
            min_derivative=self.min_derivative,
            apply_unconditional_transform=self.unconditional_transform,
        )
        t = transforms.CompositeTransform([linear, base], self.device)
        return t

    def _forward(self, inputs, context):
        """Forward pass in density estimation direction.
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        context = context # self.encoder(context)
        log_density = self.flow.log_prob(inputs, context)
        return log_density

    def forward(self, inputs, cond_inputs):
        """Forward pass to negative log likelihood (NLL).
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            cond_inputs (torch.Tensor): [N, cond_inputs_dim] tensor of cond_inputs."""
        log_density = self._forward(inputs, cond_inputs)
        loss = -torch.mean(log_density)
        return loss

    def loss(self, inputs, cond_inputs):
        """Forward pass to negative log likelihood (NLL).
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            cond_inputs (torch.Tensor): [N, cond_inputs_dim] tensor of context."""
        log_density = self._forward(inputs, cond_inputs)
        loss = -torch.mean(log_density)
        return loss

    # def sample(self, cond_inputs, num_samples):
    #     """Draw samples from the conditional flow.
    #     Args:
    #         cond_inputs (torch.Tensor): [cond_inputs_dim] tensor of conditioning info.
    #         num_samples (int): Number of samples to draw."""
    #     # cond_inputs = self.encoder(cond_inputs.unsqueeze(0)).expand(num_samples, -1)
    #     noise = self.flow._distribution.sample(1, cond_inputs)
    #     noise = noise.squeeze(1).to(self.device)
    #     samples, log_density = self.flow._transform.inverse(noise, cond_inputs)
    #     return samples, log_density

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
        samples, log_density = self.flow._transform.inverse(noise, cond_inputs)
        # samples = self.forward(inputs=noise, cond_inputs=cond_inputs, mode='inverse')[0]
        if cond_inputs is not None:
            samples = torch.cat([cond_inputs, samples], axis=1)
        if not copula:
            if transform == 'sigmoid':
                raise NotImplementedError
                # samples = sigmoid(samples)
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
                    raise NotImplementedError
                    # samples_target = sigmoid(inputs)
                    if args.conditional_copula:
                        raise NotImplementedError
                        # cond_inputs = sigmoid(cond_inputs)
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
