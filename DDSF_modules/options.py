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
            '--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
        parser.add_argument(
            '--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
        parser.add_argument(
            '--marginal', default='gamma', choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal',
            'mix_gauss_gamma', 'mix_uniform'])
        parser.add_argument(
            '--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument(
            '--obs', type=int, default=10000, help='How many data samples to generate')

        # DDSF options
        parser.add_argument(
            '--num_flow_layers_DDSF', type=int, default=10)
        parser.add_argument(
            '--num_hid_layers_DDSF', type=int, default=1)
        parser.add_argument(
            '--num_ds_dim', type=int, default=16)
        parser.add_argument(
            '--num_ds_layers', type=int, default=2)
        parser.add_argument(
            '--dimh_DDSF', type=int, default=128)

        parser.add_argument(
            '--mu', type=float, default=0, help='Mean of the Gaussian Distribution')
        parser.add_argument(
            '--var', type=float, default=1, help='Variance of the Gaussian Distribution')
        parser.add_argument(
            '--alpha', type=float, default=5, help='Parameter for the Gamma distribution')
        parser.add_argument(
            '--random_seed', type=int, default=4, help='random seed')
        parser.add_argument(
            '--exp_name', type=str, default='default_name_ddsf', help='experiment name to store plots and logs')
        parser.add_argument(
            '--figures_path', type=str, default='figures_cm', help='experiment name to store plots and logs')
        parser.add_argument(
            '--experiment_saved_models', type=str, default='saved_models')
        parser.add_argument(
            '--lr', type=float, default=1e-05)
        parser.add_argument(
            '--clip_m', type=float, default=1.0)
        parser.add_argument(
            '--beta1', type=float, default=0.9)
        parser.add_argument(
            '--beta2', type=float, default=0.999)
        parser.add_argument(
            '--low', type=float, required=False)
        parser.add_argument(
            '--high', type=float, required=False)
        parser.add_argument(
            '--early_stopping', action='store_true', help='stops training if validation stops improving')
        parser.add_argument(
            '--grid_search', action='store_true', help='grid search over hyperparameters')
        parser.add_argument(
            '--random_search', action='store_true', help='random search over hyperparameters')
        parser.add_argument(
            '--weight_decay', type=float, default=1e-09, help='adam optimizer weight decay')
        parser.add_argument(
            '--clip_grad_norm', action='store_true', help='whether to clip gradients')
        parser.add_argument(
            '--amsgrad', action='store_true', default=False, help='whether to clip gradients')

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
