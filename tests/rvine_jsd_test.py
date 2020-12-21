from utils import js_divergence
import numpy as np
import random
import torch
from RVine_modules.options import TrainOptions
# from RVine_modules.utils import gen_mv_copula
from RVine_modules.model_rvine import RVine
import os
from pathlib import Path
import pyvinecopulib as pv
from datasets.distributions import marginal_transform
from utils import normalize
from utils.visualizer import visualize_joint
import datasets
import unittest
eps = 0.0001


def gen_mv_copula(args):
    if args.mix is False:
        if args.copula == 'clayton':
            pair_copula = pv.BicopFamily.clayton
            theta = 2
        elif args.copula == 'frank':
            pair_copula = pv.BicopFamily.frank
            theta = 5
        elif args.copula == 'gumbel':
            pair_copula = pv.BicopFamily.gumbel
            theta = 5

        # Specify pair-copulas
        bicop = pv.Bicop(family=pair_copula, parameters=[theta])
        pcs = [[bicop, bicop, bicop], [bicop, bicop], [bicop]]
    else:
        bicop_1 = pv.Bicop(family=pv.BicopFamily.clayton, parameters=[2])
        bicop_2 = pv.Bicop(family=pv.BicopFamily.frank, parameters=[5])
        bicop_3 = pv.Bicop(family=pv.BicopFamily.gumbel, parameters=[5])
        pcs = [[bicop_1, bicop_2, bicop_3], [bicop_1, bicop_2], [bicop_3]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 0, 0], [4, 0, 0, 0]])

    # Set-up a vine copula
    copula = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    copula_samples = copula.simulate(n=args.obs)
    if not args.disable_marginal:
        for dim in range(copula_samples.shape[1]):
            copula_samples[:, dim] = marginal_transform(copula_samples[:, dim], marginal=args.marginal, mu=args.mu, var=args.var, alpha=args.alpha)
    assert not np.isnan(np.sum(copula_samples)), '{}'.format(copula_samples[np.isnan(copula_samples)])
    return torch.from_numpy(copula_samples), copula_samples.shape[1], copula


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
            copula_samples[:, dim] = marginal_transform(copula_samples[:, dim], marginal=args.marginal, mu=args.mu, var=args.var, alpha=args.alpha)
    assert not np.isnan(np.sum(copula_samples)), '{}'.format(copula_samples[np.isnan(copula_samples)])
    return torch.from_numpy(copula_samples), copula_samples.shape[1], copula


class Test_Rvine_2D(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_Rvine_2D, self).__init__(*args, **kwargs)
        # Training settings
        self.args = TrainOptions().parse(print=False)   # get training options
        self.args.exp_path = os.path.join('results', self.args.exp_name)
        self.args.figures_path = os.path.join(self.args.exp_path, self.args.figures_path)
        self.args.experiment_logs = os.path.join(self.args.exp_path, 'result_outputs')
        self.args.experiment_saved_models = os.path.join(self.args.experiment_saved_models, self.args.exp_name)
        self.args.RealNVP_part_of_CM_Flow = True
        # Create Folders
        self.args.epochs = 1
        self.args.obs = 10000
        self.disable_marginal = True
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.args.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        #self.args.exp_name = 'rvine_test_2d'
        #create_paths(self.args)
        self.theta = 2
        self.obs = 10000
        self.transform_fct = 'gaussian'
        self.copula = 'clayton'
        self.distr_2D_target = datasets.distributions.Copula_Distr(self.copula, self.theta, obs=self.obs)
        self.distr_2D_target.sampler(obs=self.obs)
        self.samples_2D_target = self.distr_2D_target.xx
        visualize_joint(self.samples_2D_target, 'tests/plots', '2D_cop_samples')
        self.rv = RVine(args=self.args, data=torch.from_numpy(self.samples_2D_target))
        self.rv.estimate_rvine()

        cond_noise = torch.Tensor(self.obs, 1).normal_()
        self.samples_2D_cop_flow = self.rv.cop_flow.sample(self.obs, transform='gaussian', cond_inputs=cond_noise, num_inputs=1).detach()
        assert torch.max(self.samples_2D_cop_flow) <= 1
        assert torch.min(self.samples_2D_cop_flow) >= 0
        self.samples_2D_cop_flow = torch.cat([self.samples_2D_cop_flow[:, 1:2], self.samples_2D_cop_flow[:, 0:1]], axis=1)
        visualize_joint(self.samples_2D_cop_flow, 'tests/plots', '2D_cop_flow_samples')

        self.samples_2D_rv = self.rv.sample(self.obs, transform=True).detach()
        assert torch.max(self.samples_2D_rv) <= 1
        assert torch.min(self.samples_2D_rv) >= 0
        visualize_joint(self.samples_2D_rv, 'tests/plots', '2D_rvine_samples')

    def test_2D_cop_flow(self):
        # Should be zero:
        X_in_p = np.array(self.rv.cop_flow.pdf_uniform(self.samples_2D_cop_flow[:, 1:2].numpy(), self.samples_2D_cop_flow[:, 0:1].numpy()))
        X_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_cop_flow.numpy()))
        Y_in_p = np.array(self.rv.cop_flow.pdf_uniform(self.samples_2D_target[:, 1:2], self.samples_2D_target[:, 0:1]))
        Y_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_target))

        print('X_in_p', X_in_p.mean())
        print('X_in_q', X_in_q.mean())
        print('Y_in_p', Y_in_p.mean())
        print('Y_in_q', Y_in_q.mean())

        print('with cop flow samples and cop flow pdf: ')
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        print(jsd_X_Y)
        self.assertTrue(jsd_X_Y >= 0)
        self.assertTrue(jsd_X_Y <= 1)

    def test_2D_rv_pdf(self):
        X_in_p = np.array(self.rv.pdf_uniform(self.samples_2D_cop_flow.numpy()))
        X_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_cop_flow.numpy()))
        Y_in_p = np.array(self.rv.pdf_uniform(self.samples_2D_target))
        Y_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_target))

        print('X_in_p', X_in_p.mean())
        print('X_in_q', X_in_q.mean())
        print('Y_in_p', Y_in_p.mean())
        print('Y_in_q', Y_in_q.mean())

        print('with cop flow samples and rv pdf: ')
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        print(jsd_X_Y)
        self.assertTrue(jsd_X_Y >= 0)
        self.assertTrue(jsd_X_Y <= 1)

    def test_2D_rv_samples(self):
        # Should be zero:
        X_in_p = np.array(self.rv.cop_flow.pdf_uniform(self.samples_2D_rv[:, 1:2].numpy(), self.samples_2D_rv[:, 0:1].numpy()))
        X_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_rv.numpy()))
        Y_in_p = np.array(self.rv.cop_flow.pdf_uniform(self.samples_2D_target[:, 1:2], self.samples_2D_target[:, 0:1]))
        Y_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_target))

        print('X_in_p', X_in_p.mean())
        print('X_in_q', X_in_q.mean())
        print('Y_in_p', Y_in_p.mean())
        print('Y_in_q', Y_in_q.mean())

        print('with rv samples and cop flow pdf: ')
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        print(jsd_X_Y)
        self.assertTrue(jsd_X_Y >= 0)
        self.assertTrue(jsd_X_Y <= 1)

    def test_2D_rv(self):
        # Should be zero:
        X_in_p = np.array(self.rv.pdf_uniform(self.samples_2D_rv.numpy()))
        X_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_rv.numpy()))
        Y_in_p = np.array(self.rv.pdf_uniform(self.samples_2D_target))
        Y_in_q = np.array(self.distr_2D_target.pdf(self.samples_2D_target))

        print('X_in_p', X_in_p.mean())
        print('X_in_q', X_in_q.mean())
        print('Y_in_p', Y_in_p.mean())
        print('Y_in_q', Y_in_q.mean())

        print('with rv samples and rv pdf: ')
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        print(jsd_X_Y)
        self.assertTrue(jsd_X_Y >= 0)
        self.assertTrue(jsd_X_Y <= 1)


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
    random_seed = 5
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    unittest.main()


    exit()
    # R-vine on 3D:
    args.exp_name = 'rvine_test_3d'
    create_paths(args)
    dataset_trn, dim, pv_cop = gen_mv_copula_3d(args)

    # Compare with true copula
    samples_2 = pv_cop.simulate(args.obs)

    rv = RVine(args=args, data=dataset_trn)
    rv.estimate_rvine()
    samples_1 = rv.sample(args.obs, transform=True) # no transform , or add change of var  change
    visualize_joint(np.concatenate([samples_1[:, 0:1], samples_1[:, 1:2]], axis=1), 'tests/plots', '2D_rvine_samples01')
    visualize_joint(np.concatenate([samples_1[:, 1:2], samples_1[:, 2:3]], axis=1), 'tests/plots', '2D_rvine_samples12')
    visualize_joint(np.concatenate([samples_1[:, 0:1], samples_1[:, 2:3]], axis=1), 'tests/plots', '2D_rvine_samples02')

    # Should be zero:
    X_in_p = np.array(rv.pdf_uniform(samples_1.numpy()))
    X_in_q = np.array(pv_cop.pdf(samples_1.numpy()))
    Y_in_p = np.array(rv.pdf_uniform(samples_2))
    Y_in_q = np.array(pv_cop.pdf(samples_2))

    print('X_in_p', X_in_p.mean())
    print('X_in_q', X_in_q.mean())
    print('Y_in_p', Y_in_p.mean())
    print('Y_in_q', Y_in_q.mean())
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    #jsd_changevar(rv.log_pdf, samples, pv_cop, samples_2, 'Comparison to true 3D copula, should be big: ')


    # R-vine on 4D:
    args.exp_name = 'rvine_test_4d'
    create_paths(args)
    dataset_trn, dim, pv_cop = gen_mv_copula(args)

    rv = RVine(args=args, data=dataset_trn)
    rv.estimate_rvine()
    samples = rv.sample(args.obs, transform=True) # no transform , or add change of var  change
    samples_2 = pv_cop.simulate(args.obs)

    jsd_changevar(rv.log_pdf, samples, pv_cop, samples_2, 'Comparison to true 4D copula, should be big: ')


    # # Should be big:
    # args.exp_name = 'rvine_test_4d_frankclayton'
    # create_paths(args)
    # # args.theta = 3
    # # args.copula = 'frank'
    # # dataset_trn_2, dim, pv_cop = gen_mv_copula(args)

    # # rv_2 = RVine(args=args, data=dataset_trn_2)
    # # rv_2.estimate_rvine()
    # # samples_2 = rv_2.sample(args.obs, transform=False) # no transform , or add change of var  change

    # # # assert torch.max(torch.exp(samples)) > 1, '{}'.format(torch.max(torch.exp(samples)))
    # # # assert torch.max(torch.exp(samples_2)) > 1, '{}'.format(torch.max(torch.exp(samples_2)))

    # # jsd_rvine(rv, samples, rv_2, samples_2, 'Trained on different data, should be bigger: ')

    # # Should be big
    # samples = rv.sample(args.obs, transform=True)
    # samples_2 = pv_cop.simulate(args.obs)

    # jsd_changevar(rv.log_pdf, samples, pv_cop, samples_2, 'Comparison to true copula, should be big: ')
