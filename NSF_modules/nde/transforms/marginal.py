
import torch
from torch.nn import Parameter

from NSF_modules.nde import transforms
from NSF_modules.nde.transforms import splines
import NSF_modules.utils as utils

import numpy as np


def _share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)


class MarginalSpline(transforms.Transform):
    def __init__(self,
                 transform_net_create_fn,
                 features,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 num_blocks=2,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
                 identity_init=False
                 ):
        super().__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if identity_init:
            self.unnormalized_widths = Parameter(torch.zeros(features, num_bins))
            self.unnormalized_heights = Parameter(torch.zeros(features, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (num_bins - 1) if tails == 'linear' else (num_bins + 1)
            self.unnormalized_derivatives = Parameter(constant * torch.ones(features,
                                                                            num_derivatives))
        else:
            self.unnormalized_widths = Parameter(torch.rand(features, num_bins))
            self.unnormalized_heights = Parameter(torch.rand(features, num_bins))

            num_derivatives = (num_bins - 1) if tails == 'linear' else (num_bins + 1)
            self.unnormalized_derivatives = Parameter(torch.rand(features, num_derivatives))

    def forward(self, inputs, context=None):
        return self.spline_transform(inputs, context, inverse=False)

    def inverse(self, inputs, context=None):
        return self.spline_transform(inputs, context, inverse=True)

    def spline_transform(self, inputs, context, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(self.unnormalized_heights, batch_size)
        unnormalized_derivatives = _share_across_batch(self.unnormalized_derivatives, batch_size)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)
