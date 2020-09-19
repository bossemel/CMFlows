import os
import numpy as np
import random
from pathlib import Path
import scipy.stats
import csv
from statsmodels.distributions.empirical_distribution import ECDF

from Parametric_modules.options_kdecopula import TrainOptions
from utils.visualizer import visualize_joint
import datasets.distributions
from utils import js_divergence
from utils.load_and_save import save_statistics, load_statistics
import rpy2.robjects as robjects
# import rpy2's package module
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
packnames = ('gsl', 'kdecopula', 'stats', 'VineCopula')

# R vector of strings

#Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

base = importr('base')
utils = importr('utils')
#utils.install_packages('kdecopula')

kdecopula = importr('kdecopula')
stats = importr('stats')
vinecopula = importr('VineCopula')
eps = 0.0001


def calc_jsd(test_dict, copula_pred, samples_pred, samples_target):
    # Samples from both distributinos
    pred_distr = scipy.stats.gaussian_kde(samples_pred.T)
    normal_distr = scipy.stats.norm(0, 1)
    samples_target = normal_distr.cdf(samples_target)
    assert np.min(samples_pred) >= 0
    assert np.max(samples_pred) <= 1

    # Define distributions
    true_cop_distr = datasets.distributions.Copula_Distr(args=args, transform=False)
    true_cop_distr = scipy.stats.gaussian_kde(samples_target.T)

    # Prob X in both distributions
    prob_X_in_p = pred_distr.pdf(samples_pred.T).T
    prob_X_in_q = true_cop_distr.pdf(samples_pred.T).T

    # Prob Y in both distributions
    prob_Y_in_q = true_cop_distr.pdf(samples_target.T).T
    prob_Y_in_p = pred_distr.pdf(samples_target.T).T

    assert not np.isnan(np.sum(prob_X_in_p))
    assert not np.isnan(np.sum(prob_X_in_q)), '%r' % (prob_X_in_q[:10])
    assert not np.isnan(np.sum(prob_Y_in_p))
    assert not np.isnan(np.sum(prob_Y_in_q)), '%r' % (prob_Y_in_q[:10])

    assert np.min(prob_X_in_p) >= 0
    assert np.min(prob_X_in_q) >= 0, '%r' % np.min(prob_X_in_q)
    assert np.min(prob_Y_in_p) >= 0
    assert np.min(prob_Y_in_q) >= 0

    divergence = js_divergence(prob_X_in_p=prob_X_in_p,
                               prob_X_in_q=prob_X_in_q,
                               prob_Y_in_p=prob_Y_in_p,
                               prob_Y_in_q=prob_Y_in_q)
    test_dict['js_divergence'] = divergence
    print('JS-Divergence: {}'.format(divergence))
    return test_dict


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return ys


def fit_copula(data):
    data = vinecopula.pobs(data)
    visualize_joint(np.array(data), args, name='input_data')
    kde = kdecopula.kdecop(data)
    return kde


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

    #

    # Turn off cude
    args.cuda = False

    # Set Seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Set up data loader
    # dataset, data_loaders, train_dataset = utils.load_data(args)
    dataset = datasets.distributions.Joint_Distr(args)
    test_obs = dataset.tst.shape[0]
    viz_obs = 100000
    dataset_2 = datasets.distributions.Joint_Distr(args)

    # Calculate JSD
    if args.error_bars:
        cop = fit_copula(dataset.trn)

        #samples = cop.random(obs)  # simulate random number
        samples = np.array(stats.simulate(cop, nsim=test_obs))

        test_dict = {}
        test_dict = calc_jsd(test_dict=test_dict, copula_pred=cop, samples_pred=samples, samples_target=dataset.tst)

        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=0)
        for ii in range(1, 10):
            cop = fit_copula(dataset.trn)

            #samples = cop.random(test_obs)  # simulate random number
            samples = np.array(stats.simulate(cop, nsim=test_obs))

            test_dict = {}
            test_dict = calc_jsd(test_dict=test_dict, copula_pred=cop, samples_pred=samples, samples_target=dataset.tst)

            # Gather test losses and save statistics
            test_losses = {key: [np.mean(value)] for key, value in
                           test_dict.items()}  # save test set metrics in dict format
            save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                            # save test set metrics on disk in .csv format
                            stats_dict=test_losses, current_epoch=0, continue_from_mode=True, test_epoch=0)
        stats_dict = load_statistics(args.experiment_logs, 'test_summary.csv')
        with open(os.path.join(args.experiment_logs, 'error_bars.csv'), 'w') as f:
            writer = csv.writer(f)
            for key in stats_dict.keys():
                if key != 'epoch':
                    float_list = np.array([float(xx) for xx in stats_dict[key]])
                    line = [key, np.mean(float_list), np.std(float_list)]
                    writer.writerow(line)
    else:
        cop = fit_copula(dataset.trn)

        #samples = cop.random(test_obs)  # simulate random number
        samples = np.array(stats.simulate(cop, nsim=viz_obs))

        # Visualize samples
        visualize_joint(samples, args, name='archmidean_samples')

        test_dict = {}
        samples = np.array(stats.simulate(cop, nsim=test_obs))

        test_dict = calc_jsd(test_dict=test_dict, copula_pred=cop, samples_pred=samples, samples_target=dataset.tst)

        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=0)
