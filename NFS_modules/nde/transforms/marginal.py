
import torch
from torch.nn import functional as F, Parameter

from NFS_modules.nde import transforms
from NFS_modules.nde.transforms import splines
from NFS_modules.nde.transforms.splines.rational_quadratic import rational_quadratic_spline



class MarginalSpline(transforms.Transform):
    def __init__(self, transform_net_create_fn,
                 features,
                 hidden_features,
                 context_features=None,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE
                 ):
        super().__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self._spline_parameters = Parameter(torch.randn(features))  # there’s actually a few parameters needed for each spline



    def forward(self, inputs, context=None):

        return rational_quadratic_spline(inputs, *self._spline_parameters,
            inverse=False, min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height, min_derivative=self.min_derivative)

    def inverse(self, inputs, context=None):
        return rational_quadratic_spline(inputs, *self._spline_parameters,
             min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height, min_derivative=self.min_derivative, inverse=True)

