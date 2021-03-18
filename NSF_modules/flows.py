import torch
import torch.nn as nn
import NSF_modules.nn as nn_
import NSF_modules.utils as utils
from NSF_modules.nde import distributions, flows, transforms
import scipy.stats
from utils.visualizer import visualize_joint
import numpy as np
from utils import js_divergence, t_m_metric_eval, gaussian_change_of_var_ND
import datasets.distributions


class ConditionalFlow(nn.Module):
    """A conditional rational quadratic neural spline flow."""
    def __init__(self, dim, context_dim, args):
        super().__init__()
        self.dim = dim
        self.num_inputs = dim
        self.context_dim = context_dim
        self.device = args.device

        if context_dim == 0 and dim == 1:
            args.flow_type = 'marg_flow'
            self.n_layers_m = args.n_layers_m
            self.hidden_units_m = args.hidden_units_m
            self.n_blocks_m = args.n_blocks_m
            self.n_bins_m = args.n_bins_m
            self.tail_bound_m = args.tail_bound_m
            self.dropout_m = args.dropout_m

        else:
            args.flow_type = 'cop_flow'
            self.n_layers_c = args.n_layers_c
            self.hidden_units_c = args.hidden_units_c
            self.n_blocks_c = args.n_blocks_c
            self.dropout_c = args.dropout_c
            self.n_bins_c = args.n_bins_c
            self.tail_bound_c = args.tail_bound_c

        self.use_batch_norm_c = args.use_batch_norm_c
        self.tails = args.tails
        self.min_bin_height = args.min_bin_height
        self.min_bin_width = args.min_bin_width
        self.min_derivative = args.min_derivative
        self.unconditional_transform = args.unconditional_transform
        self.device = args.device

        distribution = distributions.StandardNormal([dim]).to(args.device)
        transform = transforms.CompositeTransform([
            self.create_transform(ii) for ii in range(self.n_layers_c if args.flow_type == 'cop_flow'
                                                      else self.n_layers_m)], args.device)
        self.flow = flows.Flow(transform, distribution).to(args.device)

    def create_transform(self, ii):
        """Create invertible rational quadratic transformations."""
        if self.context_dim > 0:
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_mid_split_binary_mask(features=self.dim),
                transform_net_create_fn=lambda in_features, out_features:
                    nn_.ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        context_features=self.context_dim,
                        hidden_features=self.hidden_units_c,
                        num_blocks=self.n_blocks_c,
                        dropout_probability=self.dropout_c,
                        use_batch_norm=self.use_batch_norm_c,),
                tails=self.tails,
                tail_bound=self.tail_bound_c,
                num_bins=self.n_bins_c,
                apply_unconditional_transform=self.unconditional_transform,
            )
        if self.dim >= 2:
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(features=self.dim, even=(ii % 2 == 0)),
                transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=self.hidden_units_c,
                    num_blocks=self.n_blocks_c,
                    dropout_probability=self.dropout_c,
                    use_batch_norm=self.use_batch_norm_c
                ),
                tails=self.tails,
                tail_bound=self.tail_bound_c,
                num_bins=self.n_bins_c,
                apply_unconditional_transform=self.unconditional_transform
            )
        elif self.dim == 1:
            return transforms.PiecewiseRationalQuadraticCDF(
                shape=[self.dim],
                num_bins=self.n_bins_m,
                tails=self.tails,
                tail_bound=self.tail_bound_m)
        else:
            raise ValueError('unknown dimensionality')

    def _forward(self, inputs, context=None):
        """Forward pass in density estimation direction.
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        log_density = self.flow.log_prob(inputs, context)
        return log_density

    def pdf_normal(self, inputs, context=None):
        # Here: context normally distirbuted
        with torch.no_grad():
            normal_distr = scipy.stats.norm()
            if context is None:
                pdf = torch.exp(self._forward(inputs)).cpu().reshape(-1,)
            else:
                pdf = torch.exp(self._forward(inputs, context=context)).cpu().reshape(-1,) * \
                      normal_distr.pdf(context.cpu()).reshape(-1,)
            return pdf

    def pdf_uniform(self, inputs, context=None):
        with torch.no_grad():
            return gaussian_change_of_var_ND(inputs, self.pdf_normal, self.device, context=context)

    def transform_to_noise(self, inputs, context=None):
        noise, _ = self.flow._transform.forward(inputs, context=context)
        return noise

    def loss(self, inputs, context=None):
        """Forward pass to negative log likelihood (NLL).
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        log_density = self._forward(inputs, context)
        loss = -torch.mean(log_density)
        return loss

    def eval(self):
        self.flow.eval()

    def sample(self, num_samples=None, context=None, num_inputs=None, device=None):
        """Returns an output sample without transformation
        """
        if num_inputs:
            self.num_inputs = num_inputs
        if context is not None:
            num_samples = context.shape[0]
        noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        if device:
            if context is not None:
                context = context.to(device)
            noise = noise.to(device)
        samples, log_density = self.flow._transform.inverse(inputs=noise, context=context)
        if context is not None:
            samples = torch.cat([samples, context], axis=1)
        # if transform == 'gaussian':
        #     raise ValueError('transform after this function')
        #     # normal_distr = torch.distributions.normal.Normal(0, 1)
        #     # samples = normal_distr.cdf(samples)
        return samples

    def sample_copula(self, num_samples=None, context=None, num_inputs=None, device=None):
        """Returns the predicted copula (output sample with transformation)
        """
        if num_inputs:
            self.num_inputs = num_inputs
        noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        if device:
            noise = noise.to(device)
            if context is not None:
                context = context.to(device)
        samples, log_density = self.flow._transform.inverse(inputs=noise, context=context)
        if context is not None:
            samples = torch.cat([samples, context], axis=1)
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        return samples

    def jsd(self, args, num_samples=100000, device=None):
        """Returns JS-Divergence of the predicted Copula and the true Copula
        """
        assert num_samples == 100000
        with torch.no_grad():
            # Get ground truth
            true_cop_distr = datasets.distributions.Copula_Distr(args.copula, args.theta, obs_=num_samples)
            true_cop_distr.sampler(obs_=num_samples)
            samples_target_uni = true_cop_distr.xx

            # Samples from both distributions
            if args.conditional_copula:
                normal_distr = torch.distributions.normal.Normal(0, 1)
                context_normal = normal_distr.sample(sample_shape=torch.Size([num_samples, 1])).to(args.device)
                context_uni = normal_distr.cdf(context_normal).cpu()
            else:
                context_normal = None
                context_uni = None

            samples_pred_uni = self.sample_copula(num_samples=num_samples, context=context_normal
            if args.conditional_copula else None, device=args.device)
            samples_pred_viz = self.sample_copula(num_samples=num_samples, context=context_normal
            if args.conditional_copula else None, device=args.device)

            if args.conditional_copula:
                visualize_joint(samples_target_uni, args.figures_path, name='samples_target_jsd')
                visualize_joint(samples_pred_viz.cpu(), args.figures_path, name='samples_pred_jsd')
            else:
                visualize_joint(samples_target_uni, args.figures_path, name='samples_target_jsd')
                visualize_joint(samples_pred_viz.cpu(), args.figures_path, name='samples_pred_jsd')

            assert torch.max(samples_pred_uni) <= 1
            assert torch.min(samples_pred_uni) >= 0
            assert np.max(samples_target_uni) <= 1
            assert np.min(samples_target_uni) >= 0

            if args.conditional_copula:
                assert torch.max(context_uni) <= 1
                assert torch.min(context_uni) >= 0

            # Prob X in both distributions
            if args.conditional_copula:
                prob_x_in_p = self.pdf_uniform(inputs=np.array(samples_pred_uni[:, 0:1].cpu()), context=context_uni.numpy())
            else:
                prob_x_in_p = self.pdf_uniform(np.array(samples_pred_uni.cpu()))

            prob_x_in_q = true_cop_distr.pdf(samples_pred_uni.cpu().numpy())

            # Prob Y in both distributions
            if args.conditional_copula:
                prob_y_in_p = self.pdf_uniform(inputs=samples_target_uni[:, 0:1], context=samples_target_uni[:, 1:2])
                prob_y_in_q = true_cop_distr.pdf(np.concatenate([samples_target_uni, context_uni.cpu()], axis=1))
            else:
                prob_y_in_p = self.pdf_uniform(samples_target_uni)
                prob_y_in_q = true_cop_distr.pdf(samples_target_uni)

            assert np.min(prob_x_in_p) >= 0
            assert np.min(prob_x_in_q) >= 0
            assert np.min(prob_y_in_p) >= 0
            assert np.min(prob_y_in_q) >= 0

            assert prob_x_in_p.shape == (num_samples,), '{}'.format(prob_x_in_p.shape)
            assert prob_x_in_q.shape == (num_samples,)
            assert prob_y_in_p.shape == (num_samples,)
            assert prob_y_in_q.shape == (num_samples,)

            divergence = js_divergence(prob_x_in_p=prob_x_in_p,
                                       prob_x_in_q=prob_x_in_q,
                                       prob_y_in_p=prob_y_in_p,
                                       prob_y_in_q=prob_y_in_q)

            return divergence

    def t_metric_eval(self, args, num_samples, context=None, intervals=25, cm_flow=False, device=None):
        """Returns evaluation metrics for the copula marginals.
        """
        with torch.no_grad():
            # Samples from both distributions
            if args.conditional_copula:
                normal_distr = torch.distributions.normal.Normal(0, 1)
                context_normal = normal_distr.sample(sample_shape=torch.Size([num_samples, 1])).to(args.device)

            if args.conditional_copula:
                samples = self.sample_copula(num_samples=num_samples, context=context_normal, device=device).cpu().numpy()
            else:
                samples = self.sample_copula(num_samples=num_samples, device=device).cpu().numpy()
            if context is not None:
                margin_x1 = context
                margin_x2 = samples
            else:
                margin_x1 = samples[:, 0]
                margin_x2 = samples[:, 1]
            t_metric_x1, m_metric_x1 = t_m_metric_eval(margin_x1, intervals)
            t_metric_x2, m_metric_x2 = t_m_metric_eval(margin_x2, intervals)
            return t_metric_x1, m_metric_x1, t_metric_x2, m_metric_x2

