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

        # CM Options
        parser = argparse.ArgumentParser(description='PyTorch Flows')
        parser.add_argument(
            '--pretrain_models', action='store_true', help='first trains marginal flow, then copula flow')
        parser.add_argument(
            '--batch-size', type=int, default=100, help='input batch size for training')
        parser.add_argument(
            '--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
        parser.add_argument(
            '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
        parser.add_argument(
            '--copula', default='clayton', help='gaussian | tdistr | clayton | frank | gumbel')
        parser.add_argument(
            '--marginal_1', default='gaussian', choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'bimodal_gaussian'], help='marginal in first dimension')
        parser.add_argument(
            '--marginal_2', default='gaussian', choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'bimodal_gaussian'], help='marginal in second dimension')
        parser.add_argument(
            '--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument(
            '--obs', type=int, default=3000, help='How many data samples to generate')
        parser.add_argument(
            '--tau', type=float, required=False, help='tau to use for copula sampling')
        parser.add_argument(
            '--theta', type=float, required=False, help='theta for copula sampling')
        parser.add_argument(
            '--random_seed', type=int, default=58093, help='random seed')
        parser.add_argument(
            '--df', type=int, required=False, help='degrees of freedom for student-t copula')
        parser.add_argument(
            '--alpha', type=float, required=False, help='alpha for gamma distribution')
        parser.add_argument(
            '--exp_name', type=str, default='default_name_cm', help='experiment name to store plots and logs')
        parser.add_argument(
            '--figures_path', type=str, default='figures_cm', help='experiment name to store plots and logs')
        parser.add_argument(
            '--plot_frequ', type=int, default=10, help='save plots every x epochs')
        parser.add_argument(
            '--experiment_saved_models', type=str, default='saved_models')
        parser.add_argument(
            '--clip_grad_norm', action='store_true', default=False, help='whether to clip gradients')
        parser.add_argument(
            '--weight_decay', type=int, default=0, help='adam optimizer weight decay')

        # Options RealNVP
        parser.add_argument(
            '--num_hidden_RealNVP', type=int, default=32, help='number of hidden units')
        parser.add_argument(
            '--early_stopping', action='store_true', default=False, help='stops training after 30 unsuccessfull epochs')
        parser.add_argument(
            '--num-blocks', type=int, default=8, help='number of invertible blocks (default: 5)')
        parser.add_argument(
            '--transform_fct', type=str, default='gaussian', help='kind of transformation function before and after RealNVP, one of [sigmoid | gaussian]')

        # Options DDSF
        parser.add_argument(
            '--num_flow_layers_DDSF', type=int, default=5)
        parser.add_argument(
            '--num_hid_layers_DDSF', type=int, default=1)
        parser.add_argument(
            '--num_ds_dim', type=int, default=16)
        parser.add_argument(
            '--num_ds_layers', type=int, default=2)
        parser.add_argument(
            '--dimh_DDSF', type=int, default=128)
        parser.add_argument(
            '--amsgrad', type=int, default=0)
        parser.add_argument(
            '--polyak', type=float, default=0.0)
        parser.add_argument(
            '--clip', type=float, default=5.0)
        parser.add_argument(
            '--beta1', type=float, default=0.9)
        parser.add_argument(
            '--beta2', type=float, default=0.999)
        parser.add_argument(
            '--mu', type=float, required=False, help='mu for marginal gaussian distribution')
        parser.add_argument(
            '--var', type=float, required=False, help='var for marginal gaussian distribution')

        # Options CM Flow
        parser.add_argument(
            '--train_cm_flow', action='store_true', help='whether to train combined CM Flow')
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
