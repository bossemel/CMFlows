import argparse


class TrainOptions():
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch Flows')
        parser.add_argument(
            '--copula', default='clayton', choices=['clayton', 'frank', 'gumbel'])
        parser.add_argument(
            '--batch-size', type=int, default=100, help='input batch size for training')
        parser.add_argument(
            '--epochs', type=int, default=100, help='number of epochs to train')
        parser.add_argument(
            '--lr', type=float, default=1e-05, help='learning rate')
        parser.add_argument(
            '--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument(
            '--obs', type=int, default=10000, help='How many data samples to generate')
        parser.add_argument(
            '--tau', type=float, required=False, help='tau to use for copula sampling')
        parser.add_argument(
            '--theta', type=float, default=2, help='theta for copula sampling')
        parser.add_argument(
            '--random_seed', type=int, default=58093, help='random seed')
        parser.add_argument(
            '--df', type=int, required=False, help='degrees of freedom for student-t copula')
        parser.add_argument(
            '--early_stopping', action='store_true', default=False, help='stops training after 30 unsuccessfull epochs')
        parser.add_argument(
            '--transform_fct', type=str, default='gaussian', help='kind of transformation function before and after RealNVP, one of [sigmoid | gaussian]')
        parser.add_argument(
            '--exp_name', type=str, default='default_name_nsf', help='experiment name to store plots and logs')
        parser.add_argument(
            '--figures_path', type=str, default='figures', help='experiment name to store plots and logs')
        parser.add_argument(
            '--experiment_saved_models', type=str, default='saved_models')
        parser.add_argument(
            '--grid_search', action='store_true', help='grid search over hyperparameters')
        parser.add_argument(
            '--random_search', action='store_true', help='random search over hyperparameters')
        parser.add_argument(
            '--weight_decay', type=int, default=0, help='adam optimizer weight decay')
        parser.add_argument(
            '--beta1', type=float, default=0.9)
        parser.add_argument(
            '--beta2', type=float, default=0.999)
        parser.add_argument(
            '--conditional_copula', action='store_true')
        parser.add_argument(
            '--amsgrad', action='store_false', default=True, help='whether to clip gradients')
        parser.add_argument(
            '--error_bars', action='store_true', help='calculate mean and std over 10 experiments')
        parser.add_argument(
            '--clip_grad_norm', action='store_false', default=True, help='whether to clip gradients')


        parser.add_argument(
            '--n_layers', type=int, default=10, help='Number of spline layers in flow')
        parser.add_argument(
            '--hidden_units', type=int, default=256, help='Number of hidden units in spline layer')
        parser.add_argument(
            '--n_blocks', type=int, default=1, help='Number of residual blocks in each spline layer')
        parser.add_argument(
            '--tail_bound', type=float, default=5., help='Bounds of spline region')
        parser.add_argument(
            '--tails', type=str, default='linear', help='Function type outside spline region')
        parser.add_argument(
            '--n_bins', type=int, default=5, help='Number of bins in piecewise spline transform')
        parser.add_argument(
            '--min_bin_height', type=float, default=1e-3, help='Minimum bin height of piecewise transform')
        parser.add_argument(
            '--min_bin_width', type=float, default=1e-3, help='Minimum bin width of piecewise transform')
        parser.add_argument(
            '--min_derivative', type=float, default=1e-3, help='Minimum derivative at bin edges')
        parser.add_argument(
            '--dropout', type=float, default=0.1, help='Dropout probability in flow')
        parser.add_argument(
            '--use_batch_norm', type=int, default=1, help='Use batch norm in spline layers')
        parser.add_argument(
            '--unconditional_transform', type=int, default=0, help='Unconditionally transform identity features')
        parser.add_argument(
            '--wavelength_threshold', type=float, default=1290., help='Wavelength threshold between blue/red sides')
        parser.add_argument(
            '--subsample', type=int, default=3, help='Subsample spectra for dimensionality reduction')
        parser.add_argument(
            '--log_transform', type=int, default=0, help='Log transform spectra before standardization')
        parser.add_argument(
            '--standardize', type=int, default=1, help='Standardize spectra (by wavelength)')
        parser.add_argument(
            '--drop_outliers', type=int, default=1, help='Drop spectra with outlying flux.')
        parser.add_argument(
            '--learning_rate', type=float, default=2.5e-4, help='Initial learning rate during annealing')
        parser.add_argument(
            '--min_learning_rate', type=float, default=1e-8, help='Minimum learning rate during annealing')
        parser.add_argument(
            '--anneal_period', type=int, default=10000, help='Learning rate annealing period')
        parser.add_argument(
            '--anneal_mult', type=int, default=2, help='Warm restart period multiplier')
        parser.add_argument(
            '--n_restarts', type=int, default=3, help='Number of annealing restarts')

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
            parser = self.initialize(parser)

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

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)
        self.opt = opt
        return self.opt
