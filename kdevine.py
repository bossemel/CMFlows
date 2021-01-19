import os
import numpy as np
import random
from pathlib import Path
import csv

from KDE_modules.options_kdevine import TrainOptions
from utils.visualizer import visualize_joint
from utils import js_divergence
from utils.load_and_save import save_statistics, load_statistics
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

from RVine_modules.utils import gen_mv_copula
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})

# Import R packages
rpy2.robjects.numpy2ri.activate()  # import R's utility package
utils = rpackages.importr('utils')  # select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
packnames = ('gsl', 'VineCopula', 'kdevine')  # Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

vinecopula = importr('VineCopula')
kdevine = importr('kdevine')


def calc_jsd(test_dict, pred_distr, target_distr, samples_pred, samples_target):
    # Samples from both distributinos

    visualize_joint(samples_pred[:, :2], args.figures_path, name='samples_pred01')
    visualize_joint(samples_target[:, :2], args.figures_path, name='samples_target01')
    samples_pred[samples_pred < 0] = 0
    samples_pred[samples_pred > 1] = 1

    assert np.min(samples_pred) >= 0, '{}'.format(np.min(samples_pred))
    assert np.max(samples_pred) <= 1, '{}'.format(np.max(samples_pred))

    # Prob X in both distributions
    prob_X_in_p = np.asarray(kdevine.dkdevinecop(samples_pred, pred_distr))
    prob_X_in_q = target_distr.pdf(samples_pred)

    # Prob Y in both distributions
    prob_Y_in_q = target_distr.pdf(samples_target)
    prob_Y_in_p = np.asarray(kdevine.dkdevinecop(samples_target, pred_distr))

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
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return ys


def fit_copula(data):
    visualize_joint(np.array(data), args.figures_path, name='input_data')
    data = vinecopula.pobs(data.numpy())
    visualize_joint(np.array(data)[:, :2], args.figures_path, name='pseudo_obs')
    cop = kdevine.kdevinecop(data)
    return cop


def fit_and_evaluate(continue_from_mode, visualize):
    pred_distr = fit_copula(dataset_trn)

    if visualize:
        samples_pred = np.array(kdevine.rkdevinecop(args.viz_obs, pred_distr))
        visualize_joint(samples_pred, args.figures_path, name='archmidean_samples')

    samples_pred = np.array(kdevine.rkdevinecop(args.obs, pred_distr))
    samples_target = pv_cop.simulate(args.obs, seeds=[args.random_seed + 1])

    test_dict = {}
    test_dict = calc_jsd(test_dict=test_dict, pred_distr=pred_distr, target_distr=pv_cop,
                         samples_pred=samples_pred, samples_target=samples_target)

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   test_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=continue_from_mode, test_epoch=0)


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.RealNVP_part_of_CM_Flow = True

    # Create Folders
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)

    # Set number of obs for visualizations
    args.disable_marginal = False

    # Set Seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Set up data loader
    dataset_trn, dim, pv_cop = gen_mv_copula(args, use_seed=True)
    args.viz_obs = 10000

    # Calculate JSD
    if args.error_bars:
        fit_and_evaluate(continue_from_mode=False, visualize=True)
        for ii in range(1, 10):
            fit_and_evaluate(continue_from_mode=True, visualize=False)

        stats_dict = load_statistics(args.experiment_logs, 'test_summary.csv')
        with open(os.path.join(args.experiment_logs, 'error_bars.csv'), 'w') as f:
            writer = csv.writer(f)
            for key in stats_dict.keys():
                if key != 'epoch':
                    float_list = np.array([float(xx) for xx in stats_dict[key]])
                    line = [key, np.mean(float_list), np.std(float_list)]
                    writer.writerow(line)
    else:
        fit_and_evaluate(continue_from_mode=False, visualize=True)
