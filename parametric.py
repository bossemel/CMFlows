#import pycopula
#from pycopula.copula import ArchimedeanCopula
import copulae
import os
import numpy as np
import random
from pathlib import Path
import scipy.stats

from Parametric_modules.options import TrainOptions
from utils.visualizer import visualize_joint
import datasets.distributions
from utils import js_divergence
eps = 0.0001


def calc_jsd(copula_pred, samples_pred, samples_target):
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
    # torch.exp(self.log_density_RealNVP(samples_pred)).numpy()
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
    return divergence


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
    dataset = datasets.distributions.Joint_Distr(args)

    cop = copulae.archimedean.GumbelCopula(dim=2)
    cop.fit(dataset.trn)

    samples = cop.random(1000)  # simulate random number

    # Create Samples
    print(samples.shape)

    # Visualize samples
    visualize_joint(samples, args, name='archmidean_samples')

    # Calculate JSD
    divergence = calc_jsd(copula_pred=cop, samples_pred=samples, samples_target=dataset.tst)
    print(divergence)
