import torch.nn as nn
import torch
from torch.autograd import Variable
import scipy
from utils.various import t_m_metric_eval, sigmoid, flow_density, js_divergence
import datasets

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
        inputs_, logdets, context = inputs

        # The inputs are split and fed to each of the DDSF models
        outputs_DDSF_1, logdets_DDSF_1, __ = self.model_DDSF_1((inputs_[:, 0].reshape(-1, 1), logdets, context))
        inputs, logdets, context = inputs
        outputs_DDSF_2, logdets_DDSF_2, __ = self.model_DDSF_2((inputs_[:, 1].reshape(-1, 1), logdets, context))

        # The outputs of the DDSF are concatenated to form a bivariate distribution
        outputs_DDSFs = torch.cat((outputs_DDSF_1, outputs_DDSF_2), dim=1)

        # forward pass in RealNVP
        outputs_RealNVP, logdets_RealNVP = self.model_RealNVP(outputs_DDSFs)

        logdets = (logdets_DDSF_1.reshape(-1, 1), logdets_DDSF_2.reshape(-1, 1), logdets_RealNVP)
        outputs = (outputs_DDSF_1, outputs_DDSF_2, outputs_RealNVP)
        return outputs, logdets

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

    # def forward_copula(self, noise, mode):
    #     return self.model_RealNVP.forward(inputs=noise, mode=mode)

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
        if noise is None:
            noise = torch.Tensor(num_samples, 1).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward_RealNVP(noise, mode='inverse')[0]
        return samples

    def sample_copula(self, num_samples=None, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, 1).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward_RealNVP(noise, mode='inverse')[0]
        normal_distr = torch.distributions.normal.Normal(0, 1)
        samples = normal_distr.cdf(samples)
        samples[samples > 1] = 1 - eps
        samples[samples < 0] = 0 + eps
        return samples

    def jsd(self, inputs, transform_fct, obs=1000, cm_flow=False):
        # Define distributions
        normal_distr = scipy.stats.norm(0, 1)
        true_cop_distr = datasets.distributions.copula_distr(self.copula, self.theta)

        # Samples from both distributinos
        samples_pred = self.sample_copula(num_samples=inputs.shape[0], noise=None)
        if transform_fct == 'sigmoid':
            samples_target = torch.tensor(sigmoid(inputs))
        elif transform_fct == 'gaussian':
            samples_target = torch.tensor(normal_distr.cdf(inputs)).float()
        else:
            samples_target = torch.tensor(inputs)

        # Estimate Copula distr
        pred_distr = scipy.stats.gaussian_kde(samples_pred.T)

        # Prob X in both distributions
        prob_X_in_p = pred_distr.pdf(samples_pred.T).T
        prob_X_in_q = true_cop_distr.pdf(samples_pred.numpy())

        # Prob Y in both distributions
        prob_Y_in_q = true_cop_distr.pdf(samples_target.numpy())
        prob_Y_in_p = pred_distr.pdf(samples_target.T).T

        divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                                   prob_X_in_q=prob_X_in_q,
                                   prob_Y_in_p=prob_Y_in_p,
                                   prob_Y_in_q=prob_Y_in_q)
        return divergence

    def t_metric_eval(self, num_samples, transform_fct, intervals=25, cm_flow=False):
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

