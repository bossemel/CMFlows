import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import DDSF_modules.nn_modules as nn_
from functools import reduce


class cMADE(Module):

    def __init__(self, dim, hid_dim, context_dim, num_layers, device,
                 num_outlayers=1, activation=nn.ELU(), fixed_order=False,
                 derank=1):
        super(cMADE, self).__init__()

        oper = nn_.CWNlinear

        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = nn_.Lambda(lambda x: (activation(x[0]), x[1]))

        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers,
                           fixed_order, derank)
        ms = [m for m in map(torch.from_numpy, ms)]
        self.rx = rx

        sequels = list()
        for i in range(num_layers - 1):
            if i == 0:
                sequels.append(oper(dim, hid_dim, context_dim, device,
                                    ms[i], False))
                sequels.append(self.activation)
            else:
                sequels.append(oper(hid_dim, hid_dim, context_dim, device,
                                    ms[i], False))
                sequels.append(self.activation)

        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(hid_dim, dim * num_outlayers, context_dim, device, ms[-1])

    def forward(self, inputs):
        input_, context = inputs
        hid, _ = self.input_to_hidden((input_, context))
        out, _ = self.hidden_to_output((hid, context))
        return out.view(-1, self.dim, int(self.num_outlayers)), context

    def randomize(self):
        ms, rx = get_masks(self.dim, self.hid_dim,
                           self.num_layers, self.num_outlayers)
        for i in range(self.num_layers - 1):
            mask = torch.from_numpy(ms[i])
            if self.input_to_hidden[i * 2].mask.is_cuda:
                mask = mask.cuda()
            self.input_to_hidden[i * 2].mask.zero_().add_(mask)
        self.rx = rx


def get_mask_from_ranks(r1, r2):
    return (r2[:, None] >= r1[None, :]).astype('float32')


def get_masks_all(ds, fixed_order=False, derank=1):
    dx = ds[0]
    ms = list()
    rx = get_rank(dx, dx)
    if fixed_order:
        rx = np.sort(rx)
    r1 = rx
    if dx != 1:
        for d in ds[1:-1]:
            r2 = get_rank(dx - derank, d)
            ms.append(get_mask_from_ranks(r1, r2))
            r1 = r2
        r2 = rx - derank
        ms.append(get_mask_from_ranks(r1, r2))
    else:
        ms = [np.zeros([ds[i + 1], ds[i]]).astype('float32') for
              i in range(len(ds) - 1)]
    if derank == 1:
        assert np.all(np.diag(reduce(np.dot, ms[::-1])) == 0), 'wrong masks'

    return ms, rx


def get_masks(dim, dh, num_layers, num_outlayers, fixed_order=False, derank=1):
    ms, rx = get_masks_all([dim, ] + [dh for __ in range(num_layers - 1)] + [dim, ],
                           fixed_order, derank)
    ml = ms[-1]
    ml_ = (ml.transpose(1, 0)[:, :, None] * (np.ones(int(num_outlayers)))).reshape(
        dh, int(dim * num_outlayers)).transpose(1, 0)
    ms[-1] = ml_
    return ms, rx


def get_rank(max_rank, num_out):
    rank_out = np.array([])
    while len(rank_out) < num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = np.random.choice(max_rank, excess, False)
    rank_out = np.delete(rank_out, remove_ind)
    np.random.shuffle(rank_out)
    return rank_out.astype('float32')
