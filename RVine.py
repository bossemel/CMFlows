import torch
import torch.utils.data

import os
import numpy as np
from pathlib import Path
import random

from RVine_modules.options import TrainOptions

from RVine_modules.model_rvine import RVine

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
    dataset = torch.randn(1000, 5)

    # Initialize R-vine
    rv = RVine(args=args, data=dataset)
    rv.initialize_graph()

    # Estimate R-vine
    rv.estimate_rvine()

    # read the data and do rank transformation
    # dat = ps.read_csv("data.csv",index_col = 0)
    # dat = pd.DataFrame(np.random.randn(20, 3))

    # cp_dat = dat.rank() / (len(dat) + 1)

    # initialize R-vine object named rv

    # rv = Rvine(cp_dat)

    # sequential estimation for rv. 'structure' accepts 'r' for R-vine,
    # 'c' for C-vine and 'd' for D-vine, 'familyset' accepts list of
    # integers from 1 to 6, 'threads_num' accepts integer specifying number
    # of threads using for taking mle on edges of the same vine tree
    # simultaneously.å

    # rv.modeling(structure='r', familyset=[1, 2, 3, 4, 5, 6], threads_num=1)

    # # maximum likelihood estimation for rv. 'disp' controls the printing
    # # of ratio of progress of iterating for L-BFGS-B algorithm, 'threads_num'
    # # specifies the number of threads using for computing loglikelihood value
    # # for each edge in the same vine tree.

    # raise NotImplementedError('cm flow estimation')
    # #rv.mle(disp=False, threads_num=1)

    # # plot the R-vine structure for modeled object rv. All the vine trees will
    # # be plotted as default.

    # rv.plot()

    # # display the result of estimation on each edge. 'ndigits' controls number
    # # of decimal digits for result.

    # rv.res(ndigits=3)

    # # testing

    # rv.test()
