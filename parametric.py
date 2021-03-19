import copulae
import os
import numpy as np
import random
from pathlib import Path
import csv
import torch

from Parametric_modules.options import TrainOptions
from utils.visualizer import visualize_joint
import datasets.distributions
from utils import js_divergence
from utils.load_and_save import save_statistics, load_statistics


def calc_jsd(test_dict, pred_distr, samples_pred):
    assert np.min(samples_pred) >= 0
    assert np.max(samples_pred) <= 1

    # Define distributions
    target_distr = datasets.distributions.Copula_Distr(args.copula, args.theta, obs_=viz_obs)
    target_distr.sampler()
    samples_target = target_distr.xx

    # Prob X in both distributions
    prob_x_in_p = pred_distr.pdf(samples_pred)
    prob_x_in_q = target_distr.pdf(samples_pred)

    # Prob Y in both distributions
    prob_y_in_q = target_distr.pdf(samples_target)
    prob_y_in_p = pred_distr.pdf(samples_target)

    assert not np.isnan(np.sum(prob_x_in_p))
    assert not np.isnan(np.sum(prob_x_in_q)), '%r' % (prob_x_in_q[:10])
    assert not np.isnan(np.sum(prob_y_in_p))
    assert not np.isnan(np.sum(prob_y_in_q)), '%r' % (prob_y_in_q[:10])

    assert np.min(prob_x_in_p) >= 0
    assert np.min(prob_x_in_q) >= 0, '%r' % np.min(prob_x_in_q)
    assert np.min(prob_y_in_p) >= 0
    assert np.min(prob_y_in_q) >= 0

    divergence = js_divergence(prob_x_in_p=prob_x_in_p,
                               prob_x_in_q=prob_x_in_q,
                               prob_y_in_p=prob_y_in_p,
                               prob_y_in_q=prob_y_in_q)
    test_dict['js_divergence'] = divergence
    print('JS-Divergence: {}'.format(divergence))
    return test_dict


def fit_copula(args_, data):
    if args_.assumed_copula == 'clayton':
        cop = copulae.archimedean.ClaytonCopula(dim=2)
    elif args_.assumed_copula == 'frank':
        cop = copulae.archimedean.FrankCopula(dim=2)
    elif args_.assumed_copula == 'gumbel':
        cop = copulae.archimedean.GumbelCopula(dim=2)
    elif args_.assumed_copula == 'gaussian':
        cop = copulae.elliptical.GaussianCopula(dim=2)
    else:
        raise ValueError('Assumed copula not in list')
    cop.fit(data)
    return cop


def train_and_evaluate(continue_from_mode, visualize):
    train = load_data(args)
    pred_distr = fit_copula(args, train)

    if visualize:
        samples_pred = pred_distr.random(viz_obs)  # simulate random number
        # Visualize samples
        visualize_joint(samples_pred, args.figures_path, name='archmidean_samples')

    samples_pred = pred_distr.random(viz_obs)  # simulate random number
    test_dict = {}
    test_dict = calc_jsd(test_dict=test_dict, pred_distr=pred_distr, samples_pred=samples_pred)

    # Gather test losses and save statistics
    test_losses = {kk: [np.mean(value)] for kk, value in
                   test_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=continue_from_mode, test_epoch=0)


def load_data(args_):
    train = torch.load(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_trn'.format(args_.copula,
                                                                                                     args_.marginal_1,
                                                                                                     args_.marginal_2)))
    val = torch.load(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_val'.format(args_.copula,
                                                                                                   args_.marginal_1,
                                                                                                   args_.marginal_2)))
    train = np.concatenate([train, val], axis=0)
    return train


if __name__ == '__main__':
    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)

    # Turn off cude
    args.cuda = False

    # Set Seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Set up data loader
    # dataset, data_loaders, train_dataset = utils.load_data(args)
    # dataset = datasets.distributions.Joint_Distr(args.copula, args.marginal_1, args.marginal_2, args.theta,
    #                                              args.obs, mu=args.mu, var=args.var, alpha=args.alpha,
    #                                              random_seed=args.random_seed)
    viz_obs = 100000

    # Calculate JSD
    if args.error_bars:
        train_and_evaluate(continue_from_mode=False, visualize=True)

        for ii in range(1, 10):
            train_and_evaluate(continue_from_mode=True, visualize=False)

        stats_dict = load_statistics(args.experiment_logs, 'test_summary.csv')
        with open(os.path.join(args.experiment_logs, 'error_bars.csv'), 'w') as f:
            writer = csv.writer(f)
            for key in stats_dict.keys():
                if key != 'epoch':
                    float_list = np.array([float(xx) for xx in stats_dict[key]])
                    line = [key, np.mean(float_list), np.std(float_list)]
                    writer.writerow(line)
    else:
        train_and_evaluate(continue_from_mode=False, visualize=True)
