import pyvinecopulib as pv
import numpy as np
import torch

from datasets.distributions import marginal_transform
from utils import normalize


def gen_mv_copula(args):
    if args.mix is False:
        if args.copula == 'clayton':
            pair_copula = pv.BicopFamily.clayton
            theta = 2
        elif args.copula == 'frank':
            pair_copula = pv.BicopFamily.frank
            theta = 5
        elif args.copula == 'gumbel':
            pair_copula = pv.BicopFamily.gumbel
            theta = 5

        # Specify pair-copulas
        bicop = pv.Bicop(family=pair_copula, parameters=[theta])
        pcs = [[bicop, bicop, bicop], [bicop, bicop], [bicop]]
    else:
        bicop_1 = pv.Bicop(family=pv.BicopFamily.clayton, parameters=[2])
        bicop_2 = pv.Bicop(family=pv.BicopFamily.frank, parameters=[5])
        bicop_3 = pv.Bicop(family=pv.BicopFamily.gumbel, parameters=[5])
        pcs = [[bicop_1, bicop_2, bicop_3], [bicop_1, bicop_2], [bicop_3]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 0, 0], [4, 0, 0, 0]])

    # Set-up a vine copula
    copula = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    print(copula)
    copula_samples = copula.simulate(n=args.obs)
    for dim in range(copula_samples.shape[1]):
        copula_samples[:, dim] = normalize(marginal_transform(copula_samples[:, dim], marginal=args.marginal, args=args))
    #rvine_samples = normalize(marginal_transform(copula_samples, marginal=args.marginal, args=args))
    return torch.from_numpy(copula_samples), copula_samples.shape[1], copula
