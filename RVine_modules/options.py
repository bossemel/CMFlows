import argparse


class TrainOptions:
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self):
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch Flows')

        # CM Options
        # Architecture
        parser.add_argument(
            '--cop_flow', default='NSF', choices=['NSF', 'RealNVP'], help='which flow type to use for copula flow')
        parser.add_argument(
            '--marg_flow', default='DDSF', choices=['NSF', 'DDSF'], help='which flow type to use for marginal flow')
        parser.add_argument(
            '--use_ecdf', action='store_true')

        # Training options
        parser.add_argument(
            '--batch-size', type=int, default=128, help='input batch size for training')
        parser.add_argument(
            '--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
        parser.add_argument(
            '--lr_c', type=float, default=0.001, help='learning rate (default: 0.0001)')
        parser.add_argument(
            '--lr_m', type=float, default=0.00001, help='learning rate (default: 0.0001)')
        parser.add_argument(
            '--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument(
            '--random_seed', type=int, default=4, help='random seed')
        parser.add_argument(
            '--clip_grad_norm_c', action='store_true', help='whether to clip gradients')
        parser.add_argument(
            '--clip_grad_norm_m', action='store_true', help='whether to clip gradients')
        parser.add_argument(
            '--clip_m', type=float, default=5.0)
        parser.add_argument(
            '--clip_c', type=float, default=5.0)
        parser.add_argument(
            '--weight_decay_c', type=float, default=0.01, help='adam optimizer weight decay')
        parser.add_argument(
            '--weight_decay_m', type=float, default=1e-10, help='adam optimizer weight decay')
        parser.add_argument(
            '--error_bars', action='store_true', default=False,
            help='trains 10 times and return the standard deviation and mean of test loss')
        parser.add_argument(
            '--disable_marginal', action='store_true', default=False, help='disables marginal flow projection')
        parser.add_argument(
            '--amsgrad_c', action='store_true', default=False)
        parser.add_argument(
            '--amsgrad_m', action='store_true', default=False)

        # Dataset options
        parser.add_argument(
            '--mix', action='store_true', help='whether to create mixture R-vine as input')
        parser.add_argument(
            '--obs', type=int, default=10000, help='How many data samples to generate')
        parser.add_argument(
            '--alpha', type=float, default=5, help='alpha for gamma distribution')
        parser.add_argument(
            '--copula', default='clayton', choices=['gaussian', 'tdistr', 'clayton', 'frank', 'gumbel', 'uniform'])
        parser.add_argument(
            '--marginal', default='gamma', choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma',
                                                    'mix_lognormal', 'mix_gauss_gamma'], help='marginal distribution')
        parser.add_argument(
            '--tau', type=float, required=False, help='tau to use for copula sampling')
        parser.add_argument(
            '--theta', type=float, default=5, help='theta for copula sampling')
        parser.add_argument(
            '--df', type=int, required=False, help='degrees of freedom for student-t copula')
        parser.add_argument(
            '--mu', type=float, default=0, help='mu for marginal gaussian distribution')
        parser.add_argument(
            '--var', type=float, default=1, help='var for marginal gaussian distribution')
        parser.add_argument(
            '--low', type=float, default=0, help='lower bound for uniform distribution')
        parser.add_argument(
            '--high', type=float, default=1, help='upper bound for uniform distirbution')

        # DDSF options
        parser.add_argument(
            '--num_flow_layers_DDSF', type=int, default=9)
        parser.add_argument(
            '--num_hid_layers_DDSF', type=int, default=2)
        parser.add_argument(
            '--num_ds_dim', type=int, default=4)
        parser.add_argument(
            '--num_ds_layers', type=int, default=1)
        parser.add_argument(
            '--dimh_DDSF', type=int, default=1)

        # Options NSF - Copula estimation
        parser.add_argument(
            '--n_layers_c', type=int, default=10, help='Number of spline layers in flow')
        parser.add_argument(
            '--hidden_units_c', type=int, default=16, help='Number of hidden units in spline layer')
        parser.add_argument(
            '--n_blocks_c', type=int, default=3, help='Number of residual blocks in each spline layer')
        parser.add_argument(
            '--tail_bound_c', type=float, default=64, help='Bounds of spline region')
        parser.add_argument(
            '--tails', type=str, default='linear', help='Function type outside spline region')
        parser.add_argument(
            '--n_bins_c', type=int, default=25, help='Number of bins in piecewise spline transform')
        parser.add_argument(
            '--min_bin_height', type=float, default=1e-3, help='Minimum bin height of piecewise transform')
        parser.add_argument(
            '--min_bin_width', type=float, default=1e-3, help='Minimum bin width of piecewise transform')
        parser.add_argument(
            '--min_derivative', type=float, default=1e-3, help='Minimum derivative at bin edges')
        parser.add_argument(
            '--dropout_c', type=float, default=0.15, help='Dropout probability in flow')
        parser.add_argument(
            '--use_batch_norm_c', type=bool, default=True, help='Use batch norm in spline layers')
        parser.add_argument(
            '--unconditional_transform', type=int, default=0, help='Unconditionally transform identity features')

        # NSF Options marginal
        parser.add_argument(
            '--n_layers_m', type=int, default=10, help='Number of spline layers in flow')
        parser.add_argument(
            '--hidden_units_m', type=int, default=128, help='Number of hidden units in spline layer')
        parser.add_argument(
            '--n_blocks_m', type=int, default=2, help='Number of residual blocks in each spline layer')
        parser.add_argument(
            '--n_bins_m', type=int, default=4, help='Number of bins in piecewise spline transform')
        parser.add_argument(
            '--dropout_m', type=float, default=0.15, help='Dropout probability in flow')
        parser.add_argument(
            '--tail_bound_m', type=float, default=8, help='Bounds of spline region')
        parser.add_argument(
            '--identity_init_m', action='store_true')

        # RealNVP options
        parser.add_argument(
            '--num_hidden_RealNVP', type=int, default=32, help='number of hidden units')
        parser.add_argument(
            '--num-blocks', type=int, default=8, help='number of invertible blocks')

        # Save options
        parser.add_argument(
            '--exp_name', type=str, default='default_name_rvine', help='experiment name to store plots and logs')
        parser.add_argument(
            '--figures_path', type=str, default='figures', help='experiment name to store plots and logs')
        parser.add_argument(
            '--experiment_saved_models', type=str, default='saved_models')
        parser.add_argument(
            '--load_model', action='store_true', help='loads saved model under experiment name')
        parser.add_argument(
            '--continue_error_bars', type=int, default=0, help='from which experiment to continue error bar '
                                                               'experiments')
        parser.add_argument(
            '--data_path', type=str, default='datasets/joint_data', help='path to data')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize()

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self, print=True):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        if print:
            self.print_options(opt)
        self.opt = opt
        return self.opt
