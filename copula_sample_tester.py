import numpy as np
from datasets.distributions import sample_clayton, sample_frank, sample_gumbel, copula_pdf
import argparse
import scipy
from pathlib import Path
import os
from RealNVP_modules.utils import plot_3D


def copula_sample_test(args):
    """Plots the difference between copula samples estimated pdf and true copula pdf.
    """
    x1 = np.arange(0.01, 1, 0.01)
    x2 = np.arange(0.01, 1, 0.01)
    grid1, grid2 = np.meshgrid(x1, x2)
    grid1 = grid1.reshape(x1.shape[0] * x2.shape[0], 1)
    grid2 = grid2.reshape(x1.shape[0] * x2.shape[0], 1)
    grid = np.concatenate([grid1.reshape(-1, 1), grid2.reshape(-1, 1)], axis=1)
    if args.cop_type == 'CLAYTON':
        uu, vv = sample_clayton(80000, args.theta, uu=grid1, ww=grid2)
    if args.cop_type == 'FRANK':
        uu, vv = sample_frank(80000, args.theta, uu=grid1, ww=grid2)
    if args.cop_type == 'GUMBEL':
        uu, vv = sample_gumbel(80000, args.theta, uu=grid1, ww=grid2)
    copula_samples = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)
    cop_pdf = copula_pdf(args.cop_type, args.theta, uu=grid1, vv=grid2).reshape(-1)

    copula_samples_pdf = scipy.stats.gaussian_kde(copula_samples.T)
    copula_samples_grid = copula_samples_pdf(grid.T)

    difference = abs(copula_samples_grid - cop_pdf)

    plot_3D(figures_path, args.cop_type, grid1, grid2, cop_pdf, 'cop_pdf')
    plot_3D(figures_path, args.cop_type, grid1, grid2, copula_samples_grid, 'copula_samples')
    plot_3D(figures_path, args.cop_type, grid1, grid2, difference, 'difference')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Copula Sample Tester')
    figures_path = os.path.join('figures', 'copula_samples')
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    parser.add_argument(
        '--cop_type', type=str, required=True, help='type of copula')
    parser.add_argument(
        '--theta', type=float, required=True, help='theta for copula sampling')
    parser.add_argument(
        '--random_seed', type=int, default=58093, help='random seed')

    args = parser.parse_args()

    copula_sample_test(args)
