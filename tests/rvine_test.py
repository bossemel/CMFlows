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
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
eps = 0.0001

if __name__ == '__main__':
    args = TrainOptions().parse(print=False)   # get training options
    args.exp_name = 'rvine_uniform'
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)


    args.RealNVP_part_of_CM_Flow = True
    # Create Folders
    args.epochs = 10
    args.obs = 10000
    args.disable_marginal = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    theta = 2
    obs = args.obs
    transform_fct = 'gaussian'
    copula = 'clayton'

    distr_target = scipy.stats.uniform()
    samples_target = distr_target.rvs((obs, 4))
    assert not np.isnan(samples_target.sum())
    visualize_joint(samples_target, args.figures_path, '4D_cop_samples')

    rv = RVine(args=args, num_inputs=samples_target.shape[1])
    rv.fit(torch.from_numpy(samples_target))

    samples_pred = rv.sample(obs, transform=True).detach()
    assert torch.max(samples_pred) <= 1
    assert torch.min(samples_pred) >= 0
    visualize_joint(torch.cat([samples_pred[:, 0:1], samples_pred[:, 1:2]], axis=1), args.figures_path, '4D_rvine_samples_01')
    visualize_joint(torch.cat([samples_pred[:, 1:2], samples_pred[:, 2:3]], axis=1), args.figures_path, '4D_rvine_samples_12')
    visualize_joint(torch.cat([samples_pred[:, 2:3], samples_pred[:, 3:4]], axis=1), args.figures_path, '4D_rvine_samples_23')
    visualize_joint(torch.cat([samples_pred[:, 1:2], samples_pred[:, 3:4]], axis=1), args.figures_path, '4D_rvine_samples_13')
    visualize_joint(torch.cat([samples_pred[:, 0:1], samples_pred[:, 3:4]], axis=1), args.figures_path, '4D_rvine_samples_03')
