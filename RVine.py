import torch
import torch.utils.data

import os
import numpy as np
from pathlib import Path
import random

import pyvinecopulib as pv
import numpy as np

from RVine_modules.options import TrainOptions
from RVine_modules.model_rvine import RVine


def gen_dataset():
    # Specify pair-copulas
    bicop = pv.Bicop(pv.BicopFamily.bb1, 90, [3, 2])
    pcs = [[bicop, bicop], [bicop]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1], [2, 2, 0], [3, 0, 0]])

    # Set-up a vine copula
    cop = pv.Vinecop(mat, pcs)
    print(cop)
    u = cop.simulate(n=1000, seeds=[1, 2, 3])
    return torch.tensor(u)


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
    # dataset, data_loaders, train_dataset = utils.load_data(args)
    # dataset = gen_dataset() #
    dataset = torch.randn(1000, 3)

    # Initialize R-vine
    rv = RVine(args=args, data=dataset)
    rv.initialize_graph()

    # Estimate R-vine
    rv.estimate_rvine()

    # Sample Copula
    rv.sample_multivariate_copula()

    # # plot the R-vine structure for modeled object rv. All the vine trees will
    # # be plotted as default.

    # rv.plot()

    # # display the result of estimation on each edge. 'ndigits' controls number
    # # of decimal digits for result.

    # rv.res(ndigits=3)

    # # testing

    # rv.test()
