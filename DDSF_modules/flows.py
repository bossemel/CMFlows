import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from DDSF_modules.nn_modules import log
from torch.autograd import Variable
import DDSF_modules.iaf_modules as iaf_modules
import numpy as np
from DDSF_modules import nn_modules as nn_, utils
import math


class MAF(nn.Sequential):
    """ A sequential container for DDSFs.
    """
    def __init__(self, args, *modules):
        super(MAF, self).__init__(*modules)
        self.clip = args.clip
        self.device = args.device
        self.args = args

    def log_density(self, inputs, logdets=None, context=None):
        """Returns log of target density of the Flow

        Params:
            inputs: target distribution samples

        Returns:
            log density of the model
        """
        self.n = inputs.shape[0]
        self.clip = self.args.clip
        self.context = Variable(torch.FloatTensor(self.n, 1).zero_())
        self.logdets = Variable(torch.FloatTensor(self.n).zero_())
        self.context.to(self.device)
        self.logdets.to(self.device)
        logdets = self.logdets if logdets is None else logdets
        context = self.context if context is None else context
        u, log_jacob, __ = self((inputs, logdets, context))

        log_jacob = log_jacob.reshape(-1, 1)
        # normal distr:
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi))
        # uniform distr:
        # log_probs = Variable(torch.zeros(u.shape)) + math.log(0.5)
        # log_probs[u > 2] = -1 / 0.1
        # log_probs[u < 0] = -1 / 0.1
        # print(log_probs + log_jacob.reshape(-1, 1))
        return log_probs.reshape(-1, 1) + log_jacob.reshape(-1, 1)

    def loss(self, x):
        """Loss is negative log density
        """
        return - self.log_density(x)

    def clip_grad_norm(self):
        """Performs gradient clipping
        """
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)


class BaseFlow(Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class LinearFlow(BaseFlow):

    def __init__(self, dim, context_dim,
                 oper=nn_.ResLinear, realify=nn_.softplus):
        super(LinearFlow, self).__init__()
        self.realify = realify

        self.dim = dim
        self.context_dim = context_dim

        if type(dim) is int:
            dim_ = dim
        else:
            dim_ = np.prod(dim)

        self.mean = oper(context_dim, dim_)
        self.lstd = oper(context_dim, dim_)

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.mean, nn_.ResLinear):
            self.mean.dot_01.scale.data.uniform_(-0.001, 0.001)
            self.mean.dot_h1.scale.data.uniform_(-0.001, 0.001)
            self.mean.dot_01.bias.data.uniform_(-0.001, 0.001)
            self.mean.dot_h1.bias.data.uniform_(-0.001, 0.001)
            self.lstd.dot_01.scale.data.uniform_(-0.001, 0.001)
            self.lstd.dot_h1.scale.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1 - nn_.delta) - 1) * 0.5
                self.lstd.dot_01.bias.data.uniform_(inv - 0.001, inv + 0.001)
                self.lstd.dot_h1.bias.data.uniform_(inv - 0.001, inv + 0.001)
            else:
                self.lstd.dot_01.bias.data.uniform_(-0.001, 0.001)
                self.lstd.dot_h1.bias.data.uniform_(-0.001, 0.001)
        elif isinstance(self.mean, nn.Linear):
            self.mean.weight.data.uniform_(-0.001, 0.001)
            self.mean.bias.data.uniform_(-0.001, 0.001)
            self.lstd.weight.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1 - nn_.delta) - 1) * 0.5
                self.lstd.bias.data.uniform_(inv - 0.001, inv + 0.001)
            else:
                self.lstd.bias.data.uniform_(-0.001, 0.001)

    def forward(self, inputs):
        x, logdet, context = inputs
        mean = self.mean(context)
        lstd = self.lstd(context)
        std = self.realify(lstd)

        if type(self.dim) is int:
            x_ = mean + std * x
        else:
            size = x.size()
            x_ = mean.view(size) + std.view(size) * x
        logdet_ = nn_.sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context


class DenseSigmoidFlow(BaseFlow):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x, dim=3)
        self.act_u = lambda x: nn_.softmax(x, dim=3)

        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, logdet, dsparams):
        inv = np.log(np.exp(1 - nn_.delta) - 1)
        ndim = self.hidden_dim
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)
        x_pre_clipped = x_pre * (1 - nn_.delta) + nn_.delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(pre_w, dim=3) + \
            nn_.logsigmoid(pre_sigm[:, :, None, :]) + \
            nn_.logsigmoid(-pre_sigm[:, :, None, :]) + log(a[:, :, None, :])
        # n, d, d2, dh

        logj = logj[:, :, :, :, None] + F.log_softmax(pre_u, dim=3)[:, :, None, :, :]
        # n,  d, d2, dh, d1

        logj = utils.log_sum_exp(logj, 3).sum(3)
        # n, d, d2, d1

        logdet_ = logj + np.log(1 - nn_.delta) - \
            (log(x_pre_clipped) + log(-x_pre_clipped + 1))[:, :, :, None]

        logdet = utils.log_sum_exp(
            logdet_[:, :, :, :, None] + logdet[:, :, None, :, :], 3).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0

        return xnew, logdet


class IAF_DDSF(BaseFlow):

    def __init__(self, dim, hid_dim, context_dim, num_layers, device,
                 activation, fixed_order,
                 num_ds_dim, num_ds_layers, num_ds_multiplier=3):
        super(IAF_DDSF, self).__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        self.device = device

        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(dim=dim,
                                         hid_dim=hid_dim,
                                         context_dim=context_dim,
                                         num_layers=num_layers,
                                         device=device,
                                         num_outlayers=num_ds_multiplier * (hid_dim / dim) * num_ds_layers,
                                         activation=activation,
                                         fixed_order=fixed_order)

        num_dsparams = 0
        for i in range(num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = num_ds_dim
            if i == num_ds_layers - 1:
                out_dim = 1
            else:
                out_dim = num_ds_dim

            u_dim = in_dim
            w_dim = num_ds_dim
            a_dim = b_dim = num_ds_dim
            num_dsparams += u_dim + w_dim + a_dim + b_dim

            self.add_module('sf{}'.format(i),
                            DenseSigmoidFlow(in_dim,
                                             num_ds_dim,
                                             out_dim))
        if type(dim) is int:
            self.out_to_dsparams = nn.Conv1d(int(
                num_ds_multiplier * (hid_dim / dim) * int(num_ds_layers)),
                int(num_dsparams), 1)
        else:
            self.out_to_dsparams = nn.Conv1d(int(
                num_ds_multiplier * (hid_dim / dim[0]) * num_ds_layers), int(
                num_dsparams), 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)

    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        out = out.permute(0, 2, 1)
        dsparams = self.out_to_dsparams(out).permute(0, 2, 1)

        start = 0

        h = x.view(x.size(0), -1)[:, :, None].to(self.device)
        n = x.size(0)
        dim = self.dim if type(self.dim) is int else self.dim[0]
        lgd = Variable(torch.from_numpy(
            np.zeros((n, dim, 1, 1)).astype('float32'))).to(self.device)
        if self.out_to_dsparams.weight.is_cuda:
            lgd = lgd.cuda()
        for i in range(self.num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = self.num_ds_dim
            if i == self.num_ds_layers - 1:
                out_dim = 1
            else:
                out_dim = self.num_ds_dim

            u_dim = in_dim
            w_dim = self.num_ds_dim
            a_dim = b_dim = self.num_ds_dim
            end = start + u_dim + w_dim + a_dim + b_dim

            params = dsparams[:, :, start:end]
            h, lgd = getattr(self,'sf{}'.format(i))(h, lgd, params)
            start = end

        assert out_dim == 1, 'last dsf out dim should be 1'
        return h[:, :, 0], lgd[:, :, 0, 0].sum(1) + logdet.to(self.device), context.to(self.device)


class FlipFlow(BaseFlow):

    def __init__(self, dim):
        self.dim = dim
        super(FlipFlow, self).__init__()

    def forward(self, inputs_touple):
        inputs, logdet, context = inputs_touple

        dim = self.dim
        index = Variable(getattr(torch.arange(inputs.size(dim) - 1, -1, -1), (
                         'cpu', 'cuda')[inputs.is_cuda])().long())

        output = torch.index_select(inputs, dim, index)

        return output, logdet, context
