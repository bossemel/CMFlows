import pyvinecopulib as pv
import numpy as np
import torch


def gen_mv_copula(args, cop_type='clayton', mix=False):
    if mix is False:
        if cop_type == 'clayton':
            pair_copula = pv.BicopFamily.clayton
            theta = 2
        elif cop_type == 'frank':
            pair_copula = pv.BicopFamily.frank
            theta = 5
        elif cop_type == 'gumbel':
            pair_copula = pv.BicopFamily.gumbel
            theta = 5

    # Specify pair-copulas
    bicop = pv.Bicop(family=pair_copula, parameters=[theta])
    pcs = [[bicop, bicop], [bicop]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1], [2, 2, 0], [3, 0, 0]])

    # Set-up a vine copula
    cop = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    print(cop)
    u = cop.simulate(n=args.obs)
    return torch.tensor(u), u.shape[1], cop
