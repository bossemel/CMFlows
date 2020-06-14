import datasets.distributions as distributions
from DDSF_modules.DensityEstimator import DensityEstimator
import DDSF_modules.visualizer as visualizer
import argparse
import os
import numpy as np
import torch
import json
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
from DDSF_modules import nn_modules as nn_, flows, utils, optim
from tqdm import tqdm
from DDSF_modules.utils import load_data
from DDSF_modules.options import TrainOptions
import math


class MAF(object):

    def __init__(self, args):

        self.args = args
        self.__dict__.update(args.__dict__)
        self.num_hidden_units = args.num_hidden_units_DDSF

        dim = args.batch_size
        dimh = args.batch_size * 2 # args.dimh_DDSF
        num_flow_layers = args.num_flow_layers_DDSF

        act = nn.ELU()
        sequels = [nn_.SequentialFlow(
            flows.IAF_DDSF(dim=dim,
                           hid_dim=dimh,
                           context_dim=1,
                           num_layers=args.num_hidden_layers_DDSF + 1,
                           activation=act,
                           fixed_order=True,
                           device=args.device),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + [flows.LinearFlow(dim, 1), ]

        self.flow = nn.Sequential(*sequels).to(args.device)

        if self.cuda:
            self.flow = self.flow.cuda()

    def get_model(self):
        return self.flow

    def density(self, spl):
        n = spl.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_())
        lgd = Variable(torch.FloatTensor(n).zero_())
        zeros = Variable(torch.FloatTensor(spl.shape).zero_())
        if self.cuda:
            context = context.cuda()
            lgd = lgd.cuda()
            zeros = zeros.cuda()

        z, logdet, _ = self.flow((spl, lgd, context))

        losses = - utils.log_normal(z, zeros, zeros + 1.0).sum(1) - logdet
        return - losses

    def loss(self, x):
        return - self.density(x)

    # def loss(self, spl):
    #     n = spl.size(0)

    #     context = Variable(torch.FloatTensor(n, 1).zero_())
    #     lgd = Variable(torch.FloatTensor(n).zero_())
    #     zeros = Variable(torch.FloatTensor(n, self.num_hidden_units).zero_())

    #     u, log_jacob, __ = self.flow((spl, lgd, context))
    #     log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
    #         -1, keepdim=True)
    #     return - (log_probs + log_jacob).sum(-1, keepdim=True)

    def state_dict(self):
        return self.flow.state_dict()

    def load_state_dict(self, states):
        self.flow.load_state_dict(states)

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.flow.parameters(), self.clip)


# def load_maf_data(name):
#     if name == 'mnist':
#         return datasets.MNIST(logit=True, dequantize=True)

#     elif name == 'bsds300':
#         return datasets.BSDS300()

#     elif name == 'cifar10':
#         return datasets.CIFAR10(logit=True, flip=True, dequantize=True)

#     elif name == 'power':
#         return datasets.POWER()

#     elif name == 'gas':
#         return datasets.GAS()

#     elif name == 'hepmass':
#         return datasets.HEPMASS()

#     elif name == 'miniboone':
#         return datasets.MINIBOONE()

#     else:
#         raise ValueError('Unknown dataset')


def parse_args():
    desc = "MAF"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--copula', type=str, default='CLAYTON', choices=['CLAYTON', 'FRANK', 'GUMBEL'])
    parser.add_argument('--marginal', type=str, default='GAUSSIAN', choices=['GAUSSIAN'])
    parser.add_argument('--epochs', type=int, default=400,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--random_seed', type=int, default=1993,
                        help='Random seed')
    parser.add_argument('--fn', type=str, default='0',
                        help='Filename of model to be loaded')
    parser.add_argument('--to_train', type=int, default=1,
                        help='1 if to train 0 if not')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--polyak', type=float, default=0.0)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--num_flow_layers_DDSF', type=int, default=2)
    parser.add_argument('--num_hidden_layers_DDSF', type=int, default=1)
    parser.add_argument('--num_hidden_units_DDSF', type=int, default=16)
    parser.add_argument('--num_ds_dim', type=int, default=16)
    parser.add_argument('--num_ds_layers', type=int, default=1)
    parser.add_argument('--dimh_DDSF', type=int, default=64)
    parser.add_argument('--tau', type=int, required=False)
    parser.add_argument('--theta', type=int, required=False)
    parser.add_argument('--df', type=int, required=False)
    parser.add_argument('--obs', type=int, default=1000)
    parser.add_argument('--mu', type=float, required=False)
    parser.add_argument('--var', type=float, required=False)
    parser.add_argument('--transform_fct', type=str, default='sigmoid')
    parser.add_argument('--fixed_order', type=bool, default=True,
                        help='Fix the made ordering to be the given order')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    return check_args(parser.parse_args())


def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir + '_' + args.copula + '_' + args.marginal):
        os.makedirs(args.result_dir + '_' + args.copula + '_' + args.marginal)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    assert args.epochs >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    return args


class model(object):

    # patience = 30

    def __init__(self, args):

        self.__dict__.update(args.__dict__)

        # Set up data loader
        dataset, num_inputs, data_loaders = load_data(args)

        self.train_loader = data_loaders['train_loader']

        self.valid_loader = data_loaders['valid_loader']

        self.test_loader = data_loaders['test_loader']

        self.maf = MAF(args)

        # optim
        amsgrad = bool(args.amsgrad)
        polyak = args.polyak
        self.optim = optim.Adam(self.maf.flow.parameters(),
                                lr=args.lr,
                                betas=(args.beta1, args.beta2),
                                amsgrad=amsgrad,
                                polyak=polyak)

        # initialize checkpoint
        self.checkpoint = dict()
        self.checkpoint['best_val'] = float('inf')
        self.checkpoint['best_val_epoch'] = 0
        self.checkpoint['e'] = 0

    def train(self, epochs):
        optim = self.optim
        t = 0

        LOSSES = 0
        counter = 0

        for epoch in range(epochs):
            pbar = tqdm(total=len(self.train_loader.dataset))
            for batch_idx, data in tqdm(enumerate(self.train_loader)):
                if isinstance(data, list):
                    data = data[0]
                optim.zero_grad()
                data = Variable(data)
                if self.cuda:
                    data = data.cuda()

                losses = self.maf.loss(data)

                loss = losses.mean()

                LOSSES += losses.sum().data.cpu().numpy()
                counter += losses.size(0)

                loss.backward()
                self.maf.clip_grad_norm()
                optim.step()
                t += 1

                optim.swap()
                pbar.update(data.size(0))

            loss_val = self.evaluate(self.valid_loader)
            pbar.set_description('Train, Log likelihood in nats: {:.6f}' % (losses))
            print('Epoch: [%4d/%4d] train <= %.2f '
                  'valid: %.3f' %
                  (self.checkpoint['e'] + 1, epoch, LOSSES / float(counter),
                   loss_val))
            if loss_val < self.checkpoint['best_val']:
                print(' [^] Best validation loss [^] ... [saving]')
                self.checkpoint['best_val'] = loss_val
                self.checkpoint['best_val_epoch'] = self.checkpoint['e'] + 1

            LOSSES = 0
            counter = 0
            optim.swap()

            pbar.close()

    def evaluate(self, dataloader):
        LOSSES = 0
        c = 0
        for data in dataloader:
            if isinstance(data, list):
                data = data[0]
            data = Variable(data)
            if self.cuda:
                data = data.cuda()

            losses = self.maf.loss(data).data.cpu().numpy()
            LOSSES += losses.sum()
            c += losses.shape[0]
        return LOSSES / float(c)

    def save(self, fn):
        torch.save(self.maf.state_dict(), fn + '_model.pt')
        torch.save(self.optim.state_dict(), fn + '_optim.pt')
        with open(fn + '_args.txt', 'w') as out:
            out.write(json.dumps(self.args.__dict__, indent=4))
        with open(fn + '_checkpoint.txt', 'w') as out:
            out.write(json.dumps(self.checkpoint, indent=4))

    def load(self, fn):
        self.maf.load_state_dict(torch.load(fn + '_model.pt'))
        self.optim.load_state_dict(torch.load(fn + '_optim.pt'))

    def resume(self, fn):
        self.load(fn)
        self.checkpoint.update(
            json.loads(open(fn + '_checkpoint.txt', 'r').read()))


def main():
    # Training settings
    args = TrainOptions().parse()   # get training options

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # Cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    args.test_batch_size = args.batch_size

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed + 10000)

    print(args)

    print(" [*] Building model!")

    mdl = model(args)

    # launch the graph in a session
    if args.to_train:
        print(" [*] Training started!")
        mdl.train(args.epochs)
        print(" [*] Training finished!")

    print(" [**] Valid: %.4f" % mdl.evaluate(mdl.valid_loader))
    print(" [**] Test: %.4f" % mdl.evaluate(mdl.test_loader))

    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
    res = 200
    rng = [(-5, 5), (-5, 5)]
    # distr_1 = distributions.SwissRoll(0.5)
    # distr_1 = distributions.Gaussian(0.5)
    distr_1 = distributions.Copula_Joint(cop_type='CLAYTON', marginal='GAUSSIAN', tau=0.5, seed=5)
    denaf = DensityEstimator(dim=2)
    denaf.fit(distr_1, 1000)
    fig = visualizer.visualize2D(distr_1, denaf, res=res, rng=rng)
    fig.savefig('figures/swissroll_DDSF.png', format='png', bbox_inches='tight')
