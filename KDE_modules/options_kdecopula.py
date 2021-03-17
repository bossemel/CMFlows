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

        # Training Options
        parser.add_argument(
            '--error_bars', action='store_true', default=False, help='trains 10 times and return the standard deviation and mean of test loss')

        # Dataset options
        parser.add_argument(
            '--copula', default='clayton', choices=['clayton', 'frank', 'gumbel', 'independent'])
        parser.add_argument(
            '--marginal_1', default='gamma', choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal', 'mix_gauss_gamma', 'mix_uniform'], help='marginal distribution')
        parser.add_argument(
            '--marginal_2', default='gamma', choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal', 'mix_gauss_gamma', 'mix_uniform'], help='marginal distribution')
        parser.add_argument(
            '--obs', type=int, default=10000, help='How many data samples to generate')
        parser.add_argument(
            '--random_seed', type=int, default=4, help='random seed')
        parser.add_argument(
            '--theta', type=float, default=5)
        parser.add_argument(
            '--mu', type=float, default=0, help='Mean of the Gaussian Distribution')
        parser.add_argument(
            '--var', type=float, default=1, help='Variance of the Gaussian Distribution')
        parser.add_argument(
            '--alpha', type=float, default=5, help='Parameter for the Gamma distribution')

        # Save options
        parser.add_argument(
            '--exp_name', type=str, default='default_name_kdecopula', help='experiment name to store plots and logs')
        parser.add_argument(
            '--figures_path', type=str, default='figures_cm', help='experiment name to store plots and logs')
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
