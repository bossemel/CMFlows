import torch
import torch.utils.data

import os
import numpy as np
from pathlib import Path
import random
from itertools import combinations
import csv
import json

from RVine_modules.options import TrainOptions
from RVine_modules.model_rvine import RVine
from RVine_modules.load_and_save import save_rvine, load_rvine
from RVine_modules.utils import gen_mv_copula

from utils.visualizer import visualize_joint
from utils.load_and_save import save_statistics, load_statistics

import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})


def train_and_plot(visualize=True, continue_from_mode=False):
    # Initialize R-vine
    rv = RVine(args=args, num_inputs=dataset_trn.shape[1])

    if not args.error_bars:
        if not args.load_model:
            rv.fit(data=dataset_trn)
            save_rvine(args.experiment_saved_models, 'rvine_object', rv)
        else:
            load_rvine(args.experiment_saved_models, 'rvine_object', rv)
    else:
        rv.fit(data=dataset_trn)
    rv.jsd_vinecopula(args, pv_cop, num_samples=args.obs, visualize=visualize)

    test_losses = {key: [np.mean(value)] for key, value in
                   rv.results_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=continue_from_mode, test_epoch=None)
    if visualize:
        rv.plot()
        # Simulate and visualize
        samples = rv.sample(num_samples=args.viz_obs, transform=True)
        #pdf = rv.pdf_uniform(inputs=samples.numpy())
        #print(pdf[:10])

        paired_dims = combinations(list(range(samples.shape[1])), 2)

        normal_distr = torch.distributions.normal.Normal(0, 1)
        for pair in paired_dims:
            vis_samples = normal_distr.cdf(samples[:, pair])
            visualize_joint(vis_samples.numpy(), args.figures_path, name='rvines_dim{}'.format(pair), axis_1_name='X{}'.format(pair[0] + 1), axis_2_name='X{}'.format(pair[1] + 1))
            visualize_joint(dataset_trn[:, pair].numpy(), args.figures_path, name='true_distr_dim{}'.format(pair))
            visualize_joint(untransformed_samples[:, pair], args.figures_path, name='untransformed_true_distr_dim{}'.format(pair), axis_1_name='X{}'.format(pair[0] + 1), axis_2_name='X{}'.format(pair[1] + 1))


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.RealNVP_part_of_CM_Flow = True

    # Create Folders
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set up data loader
    dataset_trn, dim, pv_cop = gen_mv_copula(args)
    args.viz_obs = 100000
    untransformed_samples = pv_cop.simulate(args.viz_obs)

    # Set number of obs for visualizations
    args.conditional_copula = True

    if not args.error_bars:
        visualize_joint(dataset_trn[:, :2], args.figures_path, name='rvine_input_dataset01')
        visualize_joint(dataset_trn[:, 1:3], args.figures_path, name='rvine_input_dataset12')
        visualize_joint(dataset_trn[:, 2:4], args.figures_path, name='rvine_input_dataset23')

    # Estimate R-vine
    if not args.error_bars:
        train_and_plot(visualize=True, continue_from_mode=False)
    else:
        if args.continue_error_bars == 0:
            train_and_plot(visualize=True, continue_from_mode=False)
            for ii in range(1, 10):
                rv = RVine(args=args, data=dataset_trn)
                train_and_plot(visualize=False, continue_from_mode=True)
        else:
            for ii in range(args.continue_error_bars, 10):
                rv = RVine(args=args, data=dataset_trn)
                train_and_plot(visualize=False, continue_from_mode=True)

        # Load statistics and calculate mean and standard deviation
        stats_dict = load_statistics(args.experiment_logs, 'test_summary.csv')
        with open(os.path.join(args.experiment_logs, 'error_bars.csv'), 'w') as f:
            writer = csv.writer(f)
            for key in stats_dict.keys():
                if key != 'epoch':
                    float_list = np.array([float(xx) for xx in stats_dict[key]])
                    print('Evaluating {} experiments'.format(len(float_list)))
                    line = [key, np.mean(float_list), np.std(float_list)]
                    print('Mean: {}, Std.: {}'.format(np.mean(float_list), np.std(float_list)))
                    writer.writerow(line)
