import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

softplus_ = nn.Softplus()
delta = 1e-7


def softplus(x, delta=1e-6):
    # @TODO: find out why nn.Softplus() doesn't work
    return softplus_(x) + delta


def sum_from_one(x):
    return sum_from_one(x.sum(1)) if len(x.size()) > 2 else x.sum(1)


def logsigmoid(x):
    return -softplus(-x)


def log(x):
    return torch.log(x * 1e2) - np.log(1e2)


class SequentialFlow(nn.Sequential):

    def sample(self, n=1, context=None, **kwargs):
        dim = self[0].dim
        if isinstance(dim, int):
            dim = [dim, ]

        spl = torch.autograd.Variable(torch.FloatTensor(n,*dim).normal_())
        lgd = torch.autograd.Variable(torch.from_numpy(
            np.random.rand(n).astype('float32')))
        if context is None:
            context = torch.autograd.Variable(torch.from_numpy(
                np.zeros((n, self[0].context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.cuda()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(SequentialFlow, self).cuda()


class WNlinear(Module):

    def __init__(self, in_features, out_features,
                 bias=True, mask=None, norm=True):
        super(WNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('mask', mask)
        self.norm = norm
        self.direction = Parameter(torch.Tensor(out_features, in_features))
        self.scale = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:, None])
            weight = self.scale[:, None].mul(direction)
        else:
            weight = self.scale[:, None].mul(self.direction)
        if self.mask is not None:
            # weight = weight * getattr(self.mask,
            #                          ('cpu', 'cuda')[weight.is_cuda])()
            weight = weight * Variable(self.mask)
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class CWNlinear(Module):

    def __init__(self, in_features, out_features, context_features, device,
                 mask=None, norm=True):
        super(CWNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.register_buffer('mask', mask)
        self.norm = norm
        self.direction = Parameter(torch.zeros(int(out_features), in_features))
        self.cscale = nn.Linear(context_features, int(out_features))
        self.cbias = nn.Linear(context_features, int(out_features))
        self.reset_parameters()
        self.cscale.weight.data.normal_(0, 0.001)
        self.cbias.weight.data.normal_(0, 0.001)
        self.device = device

    def reset_parameters(self):
        self.direction.data.normal_(0, 0.001)

    def forward(self, inputs):
        input_, context = inputs
        scale = self.cscale(context)
        bias = self.cbias(context)
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:, None])
            weight = direction
        else:
            weight = self.direction
        if self.mask is not None:
            # weight = weight * getattr(self.mask,
            #                          ('cpu', 'cuda')[weight.is_cuda])()
            weight = weight * Variable(self.mask)
        return scale * F.linear(input_, weight.type('torch.FloatTensor').to(self.device), None) + bias, context

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class ResLinear(nn.Module):

    def __init__(
            self, in_features, out_features, bias=True, same_dim=False,
            activation=nn.ReLU(), oper=WNlinear):
        super(ResLinear, self).__init__()

        self.same_dim = same_dim

        self.dot_0h = oper(in_features, out_features, bias)
        self.dot_h1 = oper(out_features, out_features, bias)
        if not same_dim:
            self.dot_01 = oper(in_features, out_features, bias)

        self.activation = activation

    def forward(self, input_):
        h = self.activation(self.dot_0h(input_))
        out_nonlinear = self.dot_h1(h)
        out_skip = input_ if self.same_dim else self.dot_01(input_)
        return out_nonlinear + out_skip


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


class Lambda(nn.Module):

    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, input_):
        return self.function(input_)
