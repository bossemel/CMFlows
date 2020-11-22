import torch
import torch.nn as nn
import NSF_modules.nn as nn_
import NSF_modules.utils as utils
from NSF_modules.nde import distributions, flows, transforms
import scipy.stats
from utils.visualizer import visualize_joint
import numpy as np
from utils import js_divergence, t_m_metric_eval, gaussian_pdf_log
import datasets.distributions
import matplotlib.pyplot as plt


class ConditionalFlow(nn.Module):
    """A conditional rational quadratic neural spline flow."""
    def __init__(self, dim, context_dim, args):
        super().__init__()
        self.dim = dim
        self.num_inputs = dim
        self.context_dim = context_dim

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

        self.use_batch_norm = args.use_batch_norm
        self.tails = args.tails
        self.min_bin_height = args.min_bin_height
        self.min_bin_width = args.min_bin_width
        self.min_derivative = args.min_derivative
        self.unconditional_transform = args.unconditional_transform
        self.device = args.device

        self.base_transform_type = 'notaffine'
        distribution = distributions.StandardNormal([dim]).to(args.device) #distributions.TweakedUniform(high=0.999, low=0.001) #.to(args.device) # distributions.StandardNormal([dim]).to(args.device)
        transform = transforms.CompositeTransform([
            self.create_transform(ii) for ii in range(self.n_layers_c if args.flow_type == 'cop_flow' else self.n_layers_m)], args.device)
        self.flow = flows.Flow(transform, distribution).to(args.device)

    def create_transform(self, ii):
        """Create invertible rational quadratic transformations."""
        if self.context_dim > 0:
            linear = transforms.RandomPermutation(features=self.dim).to(self.device)
            base = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_mid_split_binary_mask(features=self.dim),
                transform_net_create_fn=lambda in_features, out_features:
                    nn_.ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        context_features=self.context_dim,
                        hidden_features=self.hidden_units_c,
                        num_blocks=self.n_blocks_c,
                        dropout_probability=self.dropout_c,
                        use_batch_norm=self.use_batch_norm,),
                tails=self.tails,
                tail_bound=self.tail_bound_c,
                num_bins=self.n_bins_c,
                min_bin_height=self.min_bin_height,
                min_bin_width=self.min_bin_width,
                min_derivative=self.min_derivative,
                apply_unconditional_transform=self.unconditional_transform,
            )
            return transforms.CompositeTransform([linear, base], self.device)
        elif self.dim == 2:
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(features=self.dim, even=(ii % 2 == 0)),
                transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=self.hidden_units_c,
                    num_blocks=self.n_blocks_c,
                    dropout_probability=self.dropout_c,
                    use_batch_norm=self.use_batch_norm
                ),
                tails=self.tails,
                tail_bound=self.tail_bound_c,
                num_bins=self.n_bins_c,
                apply_unconditional_transform=self.unconditional_transform
            )
        # elif self.dim == 1 and self.context_dim == 1:
        #     return transforms.MarginalSpline(transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
        #             in_features=in_features,
        #             out_features=out_features,
        #             context_features=self.context_dim,
        #             hidden_features=self.hidden_units_c,
        #             num_blocks=self.n_blocks_c,
        #             dropout_probability=self.dropout_c,
        #             use_batch_norm=self.use_batch_norm),
        #         features=self.dim,
        #         num_bins=self.n_bins_c,
        #         tails=self.tails,
        #         tail_bound=self.tail_bound_c,
        #         num_blocks=self.n_blocks_c)
        elif self.dim == 1:
            return transforms.MarginalSpline(transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    context_features=self.context_dim,
                    hidden_features=self.hidden_units_m,
                    num_blocks=self.n_blocks_m,
                    dropout_probability=self.dropout_m,
                    use_batch_norm=self.use_batch_norm),
                features=self.dim,
                num_bins=self.n_bins_m,
                tails=self.tails,
                tail_bound=self.tail_bound_m,
                num_blocks=self.n_blocks_m)
        else:
            raise ValueError('unknown dimensionality')

    def _forward(self, inputs, context=None):
        """Forward pass in density estimation direction.
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        log_density = self.flow.log_prob(inputs, context)
        return log_density

    # def _forward_copula(self, inputs, context=None):
    #     """Forward pass in density estimation direction.
    #     Args:
    #         inputs (torch.Tensor): [N, dim] tensor of data.
    #         context (torch.Tensor): [N, context_dim] tensor of context."""
    #     normal_distr = torch.distributions.normal.Normal(0, 1)
    #     zz = normal_distr.ppf(x)
    #     aa = self.forward(zz)
    #     bb = torch.log()
    #     log_density = self.flow.log_prob(inputs, context)
    #     log_density = _forward(normal_distr.ppf(inputs)) +
    #     return log_density

    def loss(self, inputs, cond_inputs=None):
        """Forward pass to negative log likelihood (NLL).
        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            cond_inputs (torch.Tensor): [N, cond_inputs_dim] tensor of context."""
        log_density = self._forward(inputs, cond_inputs)
        loss = -torch.mean(log_density)
        return loss

    def sample(self, num_samples=None, transform=None, cond_inputs=None, num_inputs=None, device=None):
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
        samples, log_density = self.flow._transform.inverse(inputs=noise, context=cond_inputs)
        if cond_inputs is not None:
            samples = torch.cat([cond_inputs, samples], axis=1)
        if transform == 'sigmoid':
            raise NotImplementedError
            # samples = sigmoid(samples)
        elif transform == 'gaussian':
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
        samples, log_density = self.flow._transform.inverse(inputs=noise, context=cond_inputs)
        if cond_inputs is not None:
            samples = torch.cat([samples, cond_inputs], axis=1)
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        return samples

    def jsd(self, args, inputs, cond_inputs=None, transform_fct=None, obs=1000): #, cm_flow=False):
        """Returns JS-Divergence of the predicted Copula and the true Copula
        """
        with torch.no_grad():
            #samples_target = torch.tensor(inputs)
            # Define distributions
            normal_distr = scipy.stats.norm(0, 1) #@Todo: do i need this?
            # true_cop_distr = datasets.distributions.Copula_Distr(args=args, transform=False)

            # Get true copula distribution
            true_cop_distr = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs)
            true_cop_distr.sampler(obs=10000)
            samples_target_uni = true_cop_distr.xx
            samples_target_normal = torch.tensor(scipy.stats.norm.ppf(samples_target_uni, loc=0, scale=1)).float()

            # Samples from both distributions
            # if args.flow_type == 'cop_flow':
            #     samples_pred = self.sample_copula(num_samples=inputs.shape[0], cond_inputs=cond_inputs, device=args.device)
            #     samples_pred_viz = self.sample_copula(num_samples=10000, cond_inputs=cond_inputs, device=args.device)
            #     assert torch.max(samples_pred) <= 1
            #     assert torch.min(samples_pred) >= 0
            # elif args.flow_type == 'marg_flow':
            samples_pred_norm = self.sample(num_samples=inputs.shape[0], cond_inputs=cond_inputs, transform=None, device=args.device)
            samples_pred_uni = self.sample_copula(num_samples=inputs.shape[0], cond_inputs=cond_inputs, device=args.device)
            samples_pred_viz = self.sample_copula(num_samples=10000, cond_inputs=cond_inputs, device=args.device)
            # else:
            #     raise ValueError('Unknown flow type.')

            if args.transform_fct == 'gaussian':
                normal_distr = torch.distributions.normal.Normal(0, 1)
                #samples_target_uni = normal_distr.cdf(inputs)
                if args.conditional_copula:
                    cond_inputs_uni = normal_distr.cdf(cond_inputs)
            else:
                raise NotImplementedError

            if args.conditional_copula:
                visualize_joint(torch.cat([samples_pred_uni, cond_inputs], axis=1).cpu(), args.figures_path, name='samples_target_jsd')
                visualize_joint(torch.cat([samples_pred_viz, cond_inputs], axis=1).cpu(), args.figures_path, name='samples_pred_jsd')
            else:
                visualize_joint(samples_pred_uni.cpu(), args.figures_path, name='samples_target_jsd')
                visualize_joint(samples_pred_viz.cpu(), args.figures_path, name='samples_pred_jsd')

            assert torch.max(samples_pred_uni) <= 1
            assert torch.min(samples_pred_uni) >= 0
            assert torch.max(samples_pred_norm) > 1
            assert torch.min(samples_pred_norm) < 0

            assert torch.max(samples_target_normal) > 1
            assert torch.min(samples_target_normal) < 0
            assert np.max(samples_target_uni) <= 1
            assert np.min(samples_target_uni) >= 0

            if args.conditional_copula:
                assert torch.max(cond_inputs) > 1
                assert torch.min(cond_inputs) < 0
                assert torch.max(cond_inputs_uni) <= 1
                assert torch.min(cond_inputs_uni) >= 0

            # Prob X in both distributions
            if args.conditional_copula:
                prob_X_in_p = pred_distr(samples_pred_norm, self.flow._log_prob, cond_inputs)
            else:
                prob_X_in_p = pred_distr(samples_pred_norm, self.flow._log_prob)

            # plt.clf()
            # new_x, new_y = zip(*sorted(zip(np.array(samples_pred_norm[:, 0]), prob_X_in_p)))
            # plt.plot(new_x, new_y)
            # plt.show()
            # exit()
            if args.conditional_copula:
                prob_X_in_q = true_cop_distr.pdf(samples_pred_uni.cpu().numpy())
            else:
                prob_X_in_q = true_cop_distr.pdf(samples_pred_uni.cpu().numpy())
            # plt.clf()
            # new_x, new_y = zip(*sorted(zip(np.array(samples_pred_uni[:, 0]), prob_X_in_q)))
            # plt.plot(new_x, new_y)
            # plt.show()
            # exit()
            # Prob Y in both distributions
            if args.conditional_copula:
                prob_Y_in_p = pred_distr(torch.cat([samples_target_normal.cpu(), cond_inputs.cpu()], axis=1).cpu(), self.flow._log_prob, cond_inputs)
                prob_Y_in_q = true_cop_distr.pdf(np.concatenate([samples_target_uni, cond_inputs_uni.cpu()], axis=1))
            else:
                prob_Y_in_p = pred_distr(samples_target_normal, self.flow._log_prob)
                prob_Y_in_q = true_cop_distr.pdf(samples_target_uni)

            assert np.min(prob_X_in_p) >= 0
            assert np.min(prob_X_in_q) >= 0
            assert np.min(prob_Y_in_p) >= 0
            assert np.min(prob_Y_in_q) >= 0

            assert prob_X_in_p.shape == (inputs.shape[0],), '{}'.format(prob_X_in_p.shape)
            assert prob_X_in_q.shape == (inputs.shape[0],)
            assert prob_Y_in_p.shape == (inputs.shape[0],)
            assert prob_Y_in_q.shape == (inputs.shape[0],)

            divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                       prob_X_in_q=prob_X_in_q,
                                       prob_Y_in_p=prob_Y_in_p,
                                       prob_Y_in_q=prob_Y_in_q)

            return divergence

    def t_metric_eval(self, num_samples, cond_inputs=None, transform_fct=None, intervals=25, cm_flow=False, device=None):
        """Returns evaluation metrics for the copula marginals.
        """
        with torch.no_grad():
            if cm_flow:
                if cond_inputs is not None:
                    samples = self.sample_copula(num_samples=num_samples, cond_inputs=cond_inputs, device=device).cpu().numpy()
                else:
                    samples = self.sample_copula(num_samples=num_samples, device=device).cpu().numpy()
            else:
                if cond_inputs is not None:
                    samples = self.sample(num_samples=num_samples, cond_inputs=cond_inputs, transform=transform_fct, device=device).cpu().numpy()
                else:
                    samples = self.sample(num_samples=num_samples, transform=transform_fct, device=device).cpu().numpy()
            if cond_inputs is not None:
                margin_x1 = cond_inputs
                margin_x2 = samples
            else:
                margin_x1 = samples[:, 0]
                margin_x2 = samples[:, 1]
            t_metric_x1, m_metric_x1 = t_m_metric_eval(margin_x1, intervals)
            t_metric_x2, m_metric_x2 = t_m_metric_eval(margin_x2, intervals)
            return t_metric_x1, m_metric_x1, t_metric_x2, m_metric_x2


def pred_distr(inputs, log_prob_func, context=None):
    if context is not None:
        inputs = inputs[:, 0:1]
    if context is not None:
        context = torch.tensor(context).float()
        context_log_prob = gaussian_pdf_log(inputs).reshape(-1,)
    else:
        context_log_prob = 0
    log_prob = np.array(log_prob_func(torch.tensor(inputs).float(), context=context))
    lognormal_pdf = gaussian_pdf_log(inputs)
    if context is not None:
        lognormal_pdf_2 = gaussian_pdf_log(context).reshape(-1,)
    else:
        lognormal_pdf_2 = 0

    output = log_prob + context_log_prob - lognormal_pdf.sum(axis=1) - lognormal_pdf_2

    output = np.exp(output)
    assert not np.isnan(output.sum())
    assert not np.isinf(output.sum())
    assert np.min(output) >= 0
    return output
