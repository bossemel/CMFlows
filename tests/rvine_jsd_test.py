from utils import js_divergence
import numpy as np
import random
import torch
from RVine_modules.options import TrainOptions
# from RVine_modules.utils import gen_mv_copula
from utils import gaussian_change_of_var_ND
from RVine_modules.model_rvine import RVine
import os
from pathlib import Path
import pyvinecopulib as pv
from datasets.distributions import marginal_transform
from utils import normalize
from utils.visualizer import visualize_joint
import datasets
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


def jsd_changevar(pdf_1, samples_1, pdf_2, samples_2, text):
    # Prob X in both distributions
    X_in_p = gaussian_change_of_var_ND(np.array(samples_1.cpu()), pdf_1, args.device)
    X_in_q = pdf_2.pdf(samples_1.cpu().numpy())
    print('mean X in p', X_in_p.mean())
    print('mean X in q', X_in_q.mean())

    # Prob Y in both distributions
    Y_in_p = gaussian_change_of_var_ND(samples_2, pdf_1, args.device)
    Y_in_q = pdf_2.pdf(samples_2)
    print('mean Y in p', Y_in_p.mean())
    print('mean Y in q', Y_in_q.mean())

    print(text)
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))


def jsd_rvine(pdf_1, samples_1, pdf_2, samples_2, text):
    # Should be zero:
    X_in_p = np.exp(np.array(pdf_1.log_pdf(samples_1)))
    X_in_q = np.exp(np.array(pdf_2.log_pdf(samples_1)))
    Y_in_p = np.exp(np.array(pdf_1.log_pdf(samples_2)))
    Y_in_q = np.exp(np.array(pdf_2.log_pdf(samples_2)))

    print(text)
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))


def create_paths(args):
    # Create Folders
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.RealNVP_part_of_CM_Flow = True

    # Create Folders
    create_paths(args)

    # Set Seed
    args.random_seed = 24
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    args.viz_obs = 10000
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    # R-vine on 2 dimensions
    args.exp_name = 'rvine_test_2d'
    create_paths(args)
    args.theta = 2
    args.obs = 10000
    args.transform_fct = 'gaussian'
    args.copula = 'clayton'
    copula_distr = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs)
    copula_distr.sampler(obs=10000)
    xx = copula_distr.xx

    rv = RVine(args=args, data=torch.from_numpy(xx))
    rv.estimate_rvine()
    samples = rv.sample(args.obs, transform=True)
    visualize_joint(samples, 'tests', '2D_rvine_samples')

    jsd_changevar(rv.log_pdf, samples, copula_distr, xx, 'Comparison to true 2dim copula, should be big: ')

    # R-vine on 3D:
    args.exp_name = 'rvine_test_3d'
    create_paths(args)
    dataset_trn, dim, pv_cop = gen_mv_copula_3d(args)

    # Compare with true copula
    samples_2 = pv_cop.simulate(args.obs)

    rv = RVine(args=args, data=dataset_trn)
    rv.estimate_rvine()
    samples = rv.sample(args.obs, transform=True) # no transform , or add change of var  change
    visualize_joint(np.concatenate([samples[:, 0:1], samples[:, 1:2]], axis=1), 'tests', '2D_rvine_samples01')
    visualize_joint(np.concatenate([samples[:, 1:2], samples[:, 2:3]], axis=1), 'tests', '2D_rvine_samples12')
    visualize_joint(np.concatenate([samples[:, 0:1], samples[:, 2:3]], axis=1), 'tests', '2D_rvine_samples02')

    jsd_changevar(rv.log_pdf, samples, pv_cop, samples_2, 'Comparison to true 3D copula, should be big: ')


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
