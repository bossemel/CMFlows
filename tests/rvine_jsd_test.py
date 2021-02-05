from utils import js_divergence
import numpy as np
import random
import torch
from RVine_modules.options import TrainOptions
from RVine_modules.model_rvine import RVine
import os
from pathlib import Path
import pyvinecopulib as pv
from datasets.distributions import marginal_transform
from utils import normalize
from utils.visualizer import visualize_joint
import datasets
import unittest
import scipy.stats
from RVine_modules.utils import gen_mv_copula


def gen_mv_copula_3d(args):
    bicop_1 = pv.Bicop(family=pv.BicopFamily.clayton, parameters=[2])
    bicop_2 = pv.Bicop(family=pv.BicopFamily.frank, parameters=[5])
    bicop_3 = pv.Bicop(family=pv.BicopFamily.gumbel, parameters=[5])
    pcs = [[bicop_1, bicop_2], [bicop_3]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1], [2, 2, 0], [3, 0, 0]])

    # Set-up a vine copula
    copula = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    copula_samples = copula.simulate(n=args.obs)
    if not args.disable_marginal:
        for dim in range(copula_samples.shape[1]):
            copula_samples[:, dim] = normalize(marginal_transform(copula_samples[:, dim], marginal=args.marginal, mu=args.mu, var=args.var, alpha=args.alpha))
    assert not np.isnan(np.sum(copula_samples)), '{}'.format(copula_samples[np.isnan(copula_samples)])
    return torch.from_numpy(copula_samples), copula_samples.shape[1], copula


def valid_pdf(self, r_vine, obs, num_inputs, device):
    pass

    # with torch.no_grad():
    #     samples = r_vine.sample(obs, transform=True)
    #     kde_distr = scipy.stats.gaussian_kde(samples.cpu().numpy().T)
    #     samples = r_vine.sample(obs, transform=True)
    #     prob_kde = kde_distr.pdf(samples.cpu().numpy().T)
    #     prob_rv = r_vine.pdf_uniform(samples.cpu().numpy())
    #     print('rv flow samples: ')
    #     print('prob_kde', prob_kde.mean())
    #     print('prob rv', prob_rv.mean())
    #     difference = np.abs(prob_kde.T - prob_rv).mean()
    #     print('difference: ', difference)
    #     self.assertTrue(difference <= 1)

    #     uniform_samples = torch.Tensor(10000, num_inputs).uniform_()
    #     print(uniform_samples.shape)
    #     prob_kde = kde_distr.pdf(uniform_samples.T)
    #     print(uniform_samples.shape)
    #     prob_rv = r_vine.pdf_uniform(uniform_samples.cpu().numpy())
    #     print('normal samples: ')
    #     print('prob_kde', prob_kde.mean())
    #     print('prob_rv', prob_rv.mean())
    #     difference = np.abs(prob_kde.T - prob_rv).mean()
    #     print('difference: ', difference)
    #     self.assertTrue(difference <= 1)

    #     kde_samples = kde_distr.resample(obs)
    #     prob_kde = kde_distr.pdf(kde_samples)

    #     kde_samples[kde_samples > 1] = 1
    #     kde_samples[kde_samples < 0] = 0
    #     prob_rv = r_vine.pdf_uniform(kde_samples.T)
    #     print('kde samples:' )
    #     print('prob_kde', prob_kde.mean())
    #     print('prob_rv', prob_rv.mean())
    #     difference = np.abs(prob_kde.T - prob_rv).mean()
    #     print('difference: ', difference)
    #     self.assertTrue(difference <= 1)


class Test_Rvine_2D(unittest.TestCase):

    def __init__(self, *_args, **kwargs):
        super(Test_Rvine_2D, self).__init__(*_args, **kwargs)
        # Training settings
        self.args = TrainOptions().parse(print=False)   # get training options
        self.args.exp_name = 'rvine_2D'
        self.args.exp_path = os.path.join('results', self.args.exp_name)
        self.args.figures_path = os.path.join(self.args.exp_path, self.args.figures_path)
        self.args.experiment_logs = os.path.join(self.args.exp_path, 'result_outputs')
        self.args.experiment_saved_models = os.path.join(self.args.experiment_saved_models, self.args.exp_name)
        Path(self.args.exp_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.figures_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.experiment_logs).mkdir(parents=True, exist_ok=True)
        Path(self.args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

        self.args.RealNVP_part_of_CM_Flow = True
        # Create Folders
        self.args.epochs = 1
        self.args.obs = 10000
        self.disable_marginal = True
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.args.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.theta = 2
        self.obs = self.args.obs
        self.transform_fct = 'gaussian'
        self.copula = 'clayton'
        self.distr_target = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        self.distr_target.sampler(obs=self.obs, transform=True)
        self.samples_target = self.distr_target.xx
        visualize_joint(self.samples_target, args.figures_path, 'test_2D_cop_samples')
        self.rv = RVine(args=self.args, num_inputs=self.samples_target.shape[1])
        self.rv.fit(torch.from_numpy(self.samples_target))

        cond_noise = torch.Tensor(self.obs, 1).normal_().to(self.args.device)
        self.samples_cop_flow = self.rv.cop_flow.sample(self.obs,
                                                        transform='gaussian',
                                                        context=cond_noise,
                                                        num_inputs=1,
                                                        device=self.args.device).detach().cpu()
        assert torch.max(self.samples_cop_flow) <= 1
        assert torch.min(self.samples_cop_flow) >= 0
        visualize_joint(self.samples_cop_flow.cpu(), args.figures_path, '2D_cop_flow_samples')

        self.samples_rv = self.rv.sample(self.obs, transform=False).detach()
        visualize_joint(self.samples_rv, args.figures_path, '2D_rvine_samples_no_transform')

        normal_distr = torch.distributions.normal.Normal(0, 1)
        visualize_joint(torch.cat([normal_distr.cdf(self.samples_rv[:, 0:1]), normal_distr.cdf(self.samples_rv[:, 1:2])], axis=1), args.figures_path, '2D_rvine_samples_self_transform')

        self.samples_rv = self.rv.sample(self.obs, transform=True).detach()
        assert torch.max(self.samples_rv) <= 1
        assert torch.min(self.samples_rv) >= 0
        visualize_joint(self.samples_rv, args.figures_path, '2D_rvine_samples')
        valid_pdf(self, self.rv, self.args.obs, self.samples_target.shape[1], self.args.device)

    def test_2D_rv_cop_flow(self):
        with torch.no_grad():
            print('samples rv', self.samples_rv.mean())
            print('samples cop flow', self.samples_cop_flow.mean())
            self.assertTrue(np.abs(self.samples_rv.mean() - self.samples_cop_flow.mean()) < 0.01)
            self.assertTrue(np.abs(self.samples_rv.std() - self.samples_cop_flow.std()) < 0.01)

            # Random Normal samples, PDF normal of RV and Cop flow
            dim_0 = torch.Tensor(self.obs, 1).uniform_()
            dim_1 = (torch.Tensor(self.obs, 1).uniform_()) * 0.5
            X_in_p_cop_flow = np.array(self.rv.cop_flow.pdf_uniform(inputs=dim_1.numpy(), context=dim_0.numpy()))
            X_in_p_rv = np.array(self.rv.pdf_uniform(torch.cat([dim_0, dim_1], axis=1).numpy()))
            self.assertAlmostEqual(X_in_p_cop_flow.mean(), X_in_p_rv.mean())
            print('X_in_p_rv', X_in_p_rv.mean())

            # Random uniform samples, PDF uniform of RV and Cop flow
            dim_0 = torch.Tensor(self.obs, 1).uniform_()
            dim_1 = torch.Tensor(self.obs, 1).uniform_()
            X_in_p_cop_flow = np.array(self.rv.cop_flow.pdf_uniform(inputs=dim_1.numpy(), context=dim_0.numpy()))
            X_in_p_rv = np.array(self.rv.pdf_uniform(torch.cat([dim_0, dim_1], axis=1).cpu().numpy()))
            self.assertTrue(np.abs(X_in_p_cop_flow.mean() - X_in_p_rv.mean()) < 0.01)
            print('X_in_p_rv', X_in_p_rv.mean())

            # Cop Flow samples, PDF uniform of RV and Cop flow
            cond_noise = torch.Tensor(self.obs, 1).normal_().to(self.args.device)
            samples = self.rv.cop_flow.sample(self.obs,
                                              transform='gaussian',
                                              context=cond_noise,
                                              num_inputs=1,
                                              device=self.args.device).detach().cpu()
            X_in_p_cop_flow = np.array(self.rv.cop_flow.pdf_uniform(inputs=samples[:, 1:2].numpy(), context=samples[:, 0:1].numpy()))
            X_in_p_rv = np.array(self.rv.pdf_uniform(torch.cat([samples[:, 0:1], samples[:, 1:2]], axis=1).cpu().numpy()))
            self.assertTrue(np.abs(X_in_p_cop_flow.mean() - X_in_p_rv.mean()) < 0.01)
            print('X_in_p_rv', X_in_p_rv.mean())

            # RV samples, PDF uniform of RV and Cop flow
            samples_rv = self.rv.sample(self.obs, transform=True)
            samples = samples_rv
            X_in_p_cop_flow = np.array(self.rv.cop_flow.pdf_uniform(inputs=samples[:, 1:2].numpy(), context=samples[:, 0:1].numpy()))
            X_in_p_rv = np.array(self.rv.pdf_uniform(torch.cat([samples[:, 0:1], samples[:, 1:2]], axis=1).cpu().numpy()))
            self.assertTrue(np.abs(X_in_p_cop_flow.mean() - X_in_p_rv.mean()) < 0.01)

    def test_2D_pred_target(self):
        with torch.no_grad():
            # Comparison target and prediction: rv samples and cop flow pdf
            self.distr_target.sampler(obs=self.obs)
            samples_target = self.distr_target.xx
            samples_rv = self.rv.sample(self.obs, transform=True)
            samples_pred = samples_rv
            X_in_p = np.array(self.rv.cop_flow.pdf_uniform(inputs=samples_pred[:, 1:2].numpy(), context=samples_pred[:, 0:1].numpy()))
            X_in_q = np.array(self.distr_target.pdf(samples_pred.numpy()))
            Y_in_p = np.array(self.rv.cop_flow.pdf_uniform(inputs=samples_target[:, 1:2], context=samples_target[:, 0:1]))
            Y_in_q = np.array(self.distr_target.pdf(samples_target))
            jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
            print('JSD: ', jsd_X_Y)
            self.assertTrue(jsd_X_Y >= 0)

            # Comparison target and prediction: rv samples and pdf
            self.distr_target.sampler(obs=self.obs)
            samples_target = self.distr_target.xx
            samples_rv = self.rv.sample(self.obs, transform=True)
            samples_pred = samples_rv
            X_in_p = np.array(self.rv.pdf_uniform(samples_pred.numpy()))
            X_in_q = np.array(self.distr_target.pdf(samples_pred.numpy()))
            Y_in_p = np.array(self.rv.pdf_uniform(samples_target))
            Y_in_q = np.array(self.distr_target.pdf(samples_target))
            jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
            print('JSD: ', jsd_X_Y)
            self.assertTrue(jsd_X_Y >= 0)

            # Comparison target and prediction: cop flow pdf and samples
            self.distr_target.sampler(obs=self.obs)
            samples_target = self.distr_target.xx
            cond_noise = torch.Tensor(self.obs, 1).normal_().to(self.args.device)
            samples_cop_flow = self.rv.cop_flow.sample(self.obs,
                                                       transform='gaussian',
                                                       context=cond_noise,
                                                       num_inputs=1,
                                                       device=self.args.device).detach().cpu()
            samples_pred = samples_cop_flow
            samples_pred = torch.cat([samples_pred[:, 1:2], samples_pred[:, 0:1]], axis=1)
            X_in_p = np.array(self.rv.cop_flow.pdf_uniform(inputs=samples_pred[:, 1:2].numpy(), context=samples_pred[:, 0:1].numpy()))
            X_in_q = np.array(self.distr_target.pdf(samples_pred.numpy()))
            Y_in_p = np.array(self.rv.cop_flow.pdf_uniform(inputs=samples_target[:, 1:2], context=samples_target[:, 0:1]))
            Y_in_q = np.array(self.distr_target.pdf(samples_target))
            jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
            print('JSD: ', jsd_X_Y)
            self.assertTrue(jsd_X_Y >= 0)

            # Comparison target and prediction: cop flow samples and pdf
            self.distr_target.sampler(obs=self.obs)
            samples_target = self.distr_target.xx
            cond_noise = torch.Tensor(self.obs, 1).normal_().to(self.args.device)
            samples_cop_flow = self.rv.cop_flow.sample(self.obs,
                                                       transform='gaussian',
                                                       context=cond_noise,
                                                       num_inputs=1,
                                                       device=self.args.device).detach().cpu()
            samples_pred = samples_cop_flow
            samples_pred = torch.cat([samples_pred[:, 1:2], samples_pred[:, 0:1]], axis=1)
            X_in_p = np.array(self.rv.pdf_uniform(samples_pred.numpy()))
            X_in_q = np.array(self.distr_target.pdf(samples_pred.numpy()))
            Y_in_p = np.array(self.rv.pdf_uniform(samples_target))
            Y_in_q = np.array(self.distr_target.pdf(samples_target))
            jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
            print('JSD: ', jsd_X_Y)
            self.assertTrue(jsd_X_Y >= 0)


class Test_Rvine_3D(unittest.TestCase):

    def __init__(self, *_args, **kwargs):
        super(Test_Rvine_3D, self).__init__(*_args, **kwargs)
        # Training settings
        self.args = TrainOptions().parse(print=False)   # get training options
        self.args.exp_name = 'rvine_3D'
        self.args.exp_path = os.path.join('results', self.args.exp_name)
        self.args.figures_path = os.path.join(self.args.exp_path, self.args.figures_path)
        self.args.experiment_logs = os.path.join(self.args.exp_path, 'result_outputs')
        self.args.experiment_saved_models = os.path.join(self.args.experiment_saved_models, self.args.exp_name)
        Path(self.args.exp_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.figures_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.experiment_logs).mkdir(parents=True, exist_ok=True)
        Path(self.args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

        self.args.RealNVP_part_of_CM_Flow = True
        # Create Folders
        self.args.epochs = 1
        self.args.obs = 10000
        self.args.disable_marginal = True
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.args.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.theta = 2
        self.obs = self.args.obs
        self.transform_fct = 'gaussian'
        self.copula = 'clayton'
        dataset_trn, dim, self.distr_target = gen_mv_copula_3d(self.args)
        self.samples_target = self.distr_target.simulate(self.args.obs)

        visualize_joint(self.samples_target, 'tests', '3D_cop_samples')

        self.rv = RVine(args=self.args, num_inputs=dataset_trn.shape[1])
        self.rv.fit(dataset_trn)

        cond_noise = torch.Tensor(self.obs, 1).normal_().to(self.args.device)
        self.samples_cop_flow = self.rv.cop_flow.sample(self.obs,
                                                        transform='gaussian',
                                                        context=cond_noise,
                                                        num_inputs=1,
                                                        device=self.args.device).detach().cpu()
        assert torch.max(self.samples_cop_flow) <= 1
        assert torch.min(self.samples_cop_flow) >= 0
        visualize_joint(self.samples_cop_flow.cpu(), 'tests', '3D_cop_flow_samples')

        self.samples_rv = self.rv.sample(self.obs, transform=True).detach()
        assert torch.max(self.samples_rv) <= 1
        assert torch.min(self.samples_rv) >= 0
        visualize_joint(self.samples_rv, 'tests', '3D_rvine_samples')

    def test_valid_pdf(self):
        valid_pdf(self, self.rv, self.args.obs, self.samples_target.shape[1], self.args.device)

    def test_3D_rv_cop_flow(self):
        with torch.no_grad():
            self.samples_pred = self.rv.sample(self.args.obs, transform=True) # no transform , or add change of var  change
            samples_pred = self.samples_pred
            self.samples_target = self.distr_target.simulate(self.args.obs)
            samples_target = self.samples_target.copy()

            visualize_joint(np.concatenate([samples_pred[:, 0:1], samples_pred[:, 1:2]], axis=1), 'tests', '3D_rvine_samples01')
            visualize_joint(np.concatenate([samples_pred[:, 1:2], samples_pred[:, 2:3]], axis=1), 'tests', '3D_rvine_samples12')
            visualize_joint(np.concatenate([samples_pred[:, 0:1], samples_pred[:, 2:3]], axis=1), 'tests', '3D_rvine_samples02')

            # Should be zero:
            X_in_p = np.array(self.rv.pdf_uniform(samples_pred.numpy()))
            X_in_q = np.array(self.distr_target.pdf(samples_pred.numpy()))
            Y_in_p = np.array(self.rv.pdf_uniform(samples_target))
            Y_in_q = np.array(self.distr_target.pdf(samples_target))

            print('X_in_p', X_in_p.mean())
            print('X_in_q', X_in_q.mean())
            print('Y_in_p', Y_in_p.mean())
            print('Y_in_q', Y_in_q.mean())
            jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
            print('3D jsd: ')
            print(jsd_X_Y)

            # RealNVP outputs the density directly, but not the transformation to
            # uniform marginals. Thus, an estimation with Gaussian KDE is simpler.
            pred_distr = scipy.stats.gaussian_kde(samples_pred.cpu().numpy().T)
            true_rvine = scipy.stats.gaussian_kde(samples_target.T)
            # Note, that uniform samples means the transformed samples

            # Prob X in both distributions
            prob_X_in_p = pred_distr.pdf(samples_pred.cpu().numpy().T).T
            prob_X_in_q = true_rvine.pdf(samples_pred.cpu().numpy().T).T

            # Prob Y in both distributions
            prob_Y_in_q = true_rvine.pdf(samples_target.T).T
            prob_Y_in_p = pred_distr.pdf(samples_target.T).T
            divergence_2 = js_divergence(prob_X_in_p=prob_X_in_p,
                                         prob_X_in_q=prob_X_in_q,
                                         prob_Y_in_p=prob_Y_in_p,
                                         prob_Y_in_q=prob_Y_in_q)
            print('kde jsd: ', divergence_2)

            self.assertTrue(jsd_X_Y >= 0)


class Test_Rvine_4D(unittest.TestCase):

    def __init__(self, *_args, **kwargs):
        super(Test_Rvine_4D, self).__init__(*_args, **kwargs)
        # Training settings
        self.args = TrainOptions().parse(print=False)   # get training options
        self.args.exp_name = 'rvine_4D'
        self.args.exp_path = os.path.join('results', self.args.exp_name)
        self.args.figures_path = os.path.join(self.args.exp_path, self.args.figures_path)
        self.args.experiment_logs = os.path.join(self.args.exp_path, 'result_outputs')
        self.args.experiment_saved_models = os.path.join(self.args.experiment_saved_models, self.args.exp_name)
        Path(self.args.exp_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.figures_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.experiment_logs).mkdir(parents=True, exist_ok=True)
        Path(self.args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

        self.args.RealNVP_part_of_CM_Flow = True
        # Create Folders
        self.args.epochs = 1
        self.args.obs = 10000
        self.args.disable_marginal = True
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.args.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.theta = 2
        self.obs = self.args.obs
        self.transform_fct = 'gaussian'
        self.copula = 'clayton'
        dataset_trn, dim, self.distr_target = gen_mv_copula(self.args)
        self.samples_target = self.distr_target.simulate(self.args.obs)

        visualize_joint(self.samples_target, 'tests', '4D_cop_samples')

        self.rv = RVine(args=self.args, num_inputs=self.samples_target.shape[1])
        self.rv.fit(torch.from_numpy(self.samples_target))

        cond_noise = torch.Tensor(self.obs, 1).normal_().to(self.args.device)
        self.samples_cop_flow = self.rv.cop_flow.sample(self.obs,
                                                        transform='gaussian',
                                                        context=cond_noise,
                                                        num_inputs=1,
                                                        device=self.args.device).detach().cpu()
        assert torch.max(self.samples_cop_flow) <= 1
        assert torch.min(self.samples_cop_flow) >= 0
        visualize_joint(self.samples_cop_flow.cpu(), 'tests', '4D_cop_flow_samples')

        self.samples_rv = self.rv.sample(self.obs, transform=True).detach()
        assert torch.max(self.samples_rv) <= 1
        assert torch.min(self.samples_rv) >= 0
        visualize_joint(self.samples_rv, 'tests', '4D_rvine_samples')

    def test_valid_pdf(self):
        valid_pdf(self, self.rv, self.args.obs, self.samples_target.shape[1], self.args.device)

    def test_4D_rv_cop_flow(self):
        with torch.no_grad():
            self.samples_pred = self.rv.sample(self.args.obs, transform=True) # no transform , or add change of var  change
            samples_pred = self.samples_pred
            self.samples_target = self.distr_target.simulate(self.args.obs)
            samples_target = self.samples_target.copy()

            visualize_joint(np.concatenate([samples_pred[:, 0:1], samples_pred[:, 1:2]], axis=1), 'tests', '4D_rvine_samples01')
            visualize_joint(np.concatenate([samples_pred[:, 1:2], samples_pred[:, 2:3]], axis=1), 'tests', '4D_rvine_samples12')
            visualize_joint(np.concatenate([samples_pred[:, 0:1], samples_pred[:, 2:3]], axis=1), 'tests', '4D_rvine_samples02')

            # Should be zero:
            X_in_p = np.array(self.rv.pdf_uniform(samples_pred.numpy()))
            X_in_q = np.array(self.distr_target.pdf(samples_pred.numpy()))
            Y_in_p = np.array(self.rv.pdf_uniform(samples_target))
            Y_in_q = np.array(self.distr_target.pdf(samples_target))

            print('X_in_p', X_in_p.mean())
            print('X_in_q', X_in_q.mean())
            print('Y_in_p', Y_in_p.mean())
            print('Y_in_q', Y_in_q.mean())
            jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
            print('4D jsd: ')
            print(jsd_X_Y)

            # RealNVP outputs the density directly, but not the transformation to
            # uniform marginals. Thus, an estimation with Gaussian KDE is simpler.
            pred_distr = scipy.stats.gaussian_kde(samples_pred.cpu().numpy().T)
            true_rvine = scipy.stats.gaussian_kde(samples_target.T)
            # Note, that uniform samples means the transformed samples

            # Prob X in both distributions
            prob_X_in_p = pred_distr.pdf(samples_pred.cpu().numpy().T).T
            prob_X_in_q = true_rvine.pdf(samples_pred.cpu().numpy().T).T

            # Prob Y in both distributions
            prob_Y_in_q = true_rvine.pdf(samples_target.T).T
            prob_Y_in_p = pred_distr.pdf(samples_target.T).T
            divergence_2 = js_divergence(prob_X_in_p=prob_X_in_p,
                                         prob_X_in_q=prob_X_in_q,
                                         prob_Y_in_p=prob_Y_in_p,
                                         prob_Y_in_q=prob_Y_in_q)
            print('kde jsd: ', divergence_2)

            self.assertTrue(jsd_X_Y >= 0)

if __name__ == '__main__':

    args = TrainOptions().parse(print=False)   # get training options

    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

    # R-vine on 2 dimensions
    #for random_seed in range(5):
    random_seed = 51
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    unittest.main()

