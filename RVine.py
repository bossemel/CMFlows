import torch
import torch.utils.data

import os
import numpy as np
from pathlib import Path
import random
from itertools import combinations

from RVine_modules.options import TrainOptions
from RVine_modules.model_rvine import RVine
from RVine_modules.load_and_save import save_rvine, load_rvine
from RVine_modules.eval import jsd_eval
from RVine_modules.utils import gen_mv_copula

from utils.visualizer import visualize_joint
from utils.load_and_save import save_statistics

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
    untransformed_samples = pv_cop.simulate(100000)
    # dataset_trn = torch.randn(1000, 4)

    pv_cop.plot()
    plt.show()
    # Initialize R-vine
    rv = RVine(args=args, data=dataset_trn)

    # Estimate R-vine
    if not args.load_model:
        rv.estimate_rvine()
        save_rvine(args.experiment_saved_models, 'rvine_object', rv)
    else:
        load_rvine(args.experiment_saved_models, 'rvine_object', rv)

    rv.plot()

    assert len(rv.tree_list) > 0
    # Get density estimate
    # rv.density(torch.randn(10, 4))

    # jsd_eval(args, dim, dataset_trn, pv_cop, rv)

    # Simulate Distribution
    samples = rv.sample(num_samples=100000)

    paired_dims = combinations(list(range(samples.shape[1])), 2)

    normal_distr = torch.distributions.normal.Normal(0, 1)
    for pair in paired_dims:
        vis_samples = normal_distr.cdf(samples[:, pair])
        visualize_joint(vis_samples.numpy(), args, name='rvines_dim{}'.format(pair))
        visualize_joint(dataset_trn[:, pair].numpy(), args, name='true_distr_dim{}'.format(pair))
        visualize_joint(untransformed_samples[:, pair], args, name='untransformed_true_distr_dim{}'.format(pair))

    rv.jsd_vinecopula(args, rv, pv_cop, obs=10000)

    # Save results
    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   rv.results_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=None)

    # Sample Copula
    # rv.sample_multivariate_copula()

    # # plot the R-vine structure for modeled object rv. All the vine trees will
    # # be plotted as default.

    # # display the result of estimation on each edge. 'ndigits' controls number
    # # of decimal digits for result.

    # rv.res(ndigits=3)

    # # testing

    # rv.test()
