import torch.utils.data
import os
import numpy as np
from pathlib import Path
import random
from itertools import combinations
import csv
import matplotlib
import json

from RVine_modules.options import TrainOptions
from RVine_modules.model_rvine import RVine
from RVine_modules.load_and_save import save_rvine
from RVine_modules.utils import load_mv_copula
from utils.visualizer import visualize_joint
from utils.load_and_save import save_statistics, load_statistics

matplotlib.rcParams.update({'figure.max_open_warning': 0})


def train_and_plot(visualize=True, continue_from_mode=False):
    # Set up data loader
    dataset_trn, dim, pv_cop = load_mv_copula(args)

    # Initialize R-vine
    model_ = RVine(args=args, num_inputs=dataset_trn.shape[1])

    if not args.error_bars:
        if not args.error_bars:
            visualize_joint(dataset_trn[:, :2], args.figures_path, name='rvine_input_dataset01')
            visualize_joint(dataset_trn[:, 1:3], args.figures_path, name='rvine_input_dataset12')
            visualize_joint(dataset_trn[:, 2:4], args.figures_path, name='rvine_input_dataset23')

        # if not args.load_model_:
        model_.fit(data=dataset_trn)
        save_rvine(args.experiment_saved_models, 'rvine_object', model_)
        # else:
        #     load_rvine(args.experiment_saved_model_s, 'rvine_object', model_)
    else:
        model_.fit(data=dataset_trn)
    model_.jsd_vinecopula(args, pv_cop, num_samples=args.viz_obs)

    test_losses = {kk: [np.mean(value)] for kk, value in
                   model_.results_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=continue_from_mode, test_epoch=None)
    if visualize:
        model_.plot()
        # Simulate and visualize
        samples = model_.sample(num_samples=args.viz_obs, transform=True)
        # normal_distr = torch.distributions.normal.Normal(0, 1)
        # samples = normal_distr.cdf(samples)

        paired_dims = combinations(list(range(samples.shape[1])), 2)

        for pair in paired_dims:
            viz_samples = samples[:, pair]
            visualize_joint(viz_samples.numpy(), args.figures_path, name='rvines_dim{}'.format(pair),
                            axis_1_name='X{}'.format(pair[0]), axis_2_name='X{}'.format(pair[1]))
            visualize_joint(dataset_trn[:, pair].numpy(), args.figures_path, name='true_distr_dim{}'.format(pair))


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
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    # Cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set number of obs for visualizations
    args.conditional_copula = True
    args.viz_obs = 100000

    # Estimate R-vine
    if not args.error_bars:
        train_and_plot(visualize=True, continue_from_mode=False)
    else:
        if args.continue_error_bars == 0:
            train_and_plot(visualize=True, continue_from_mode=False)
            for ii in range(1, 10):
                model = RVine(args=args, num_inputs=args.obs)
                train_and_plot(visualize=False, continue_from_mode=True)
        else:
            for ii in range(args.continue_error_bars, 10):
                model = RVine(args=args, num_inputs=args.obs)
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
