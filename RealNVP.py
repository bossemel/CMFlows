import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import os
import numpy as np
from pathlib import Path
import random

import RealNVP_modules.flows as fnn
import RealNVP_modules.utils as utils
from RealNVP_modules.options import TrainOptions
from RealNVP_modules.eval import jsd_graph

from utils.visualizer import visualize_joint
from utils.save_statistics import save_statistics
import datasets.distributions

from experiment_runner import train_val


def build_model(args):
    """Creates model.

    Params:
        args: option input arguments

    Returns: RealNVP model
    """
    num_hidden = {args.copula: args.num_hidden_RealNVP}[args.copula]
    num_inputs = 2
    modules = []

    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(args.device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    return model


def train_and_plot():
    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set up data loader
    dataset, data_loaders = utils.load_data(args)

    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.to(args.device)
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Save losses and best epoch stats and model in dictionary
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0, 'best_model': model}
    test_dict = {"test_loss": [], 'jsd_test_copula': [], 't_1': [], 't_2': [], 'm_1': [], 'm_2': []}  # initialize a statistics dict

    # Train
    model, best_dict, test_dict = train_val(current_model=model,
                                            model_name='RealNVP',
                                            args=args,
                                            data_loaders=data_loaders,
                                            dataset=dataset,
                                            transform_inputs=False,
                                            cm_flow=False)

    output_copula = model.sample(num_samples=100000, transform=args.transform_fct)
    visualize_joint(output_copula.detach().numpy(), args, name='output_copula')

    dataset = datasets.distributions.Copula_Distr(args, transform=False)
    visualize_joint(dataset.trn.x, args, name='true_{}_copula_cm'.format(args.copula))

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   test_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # Plot pointwise copula difference
    jsd_graph(args,
              best_dict['best_validation_epoch'],
              model,
              cm_flow=True)


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

    # Cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")


    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set up data loader
    dataset, data_loaders = utils.load_data(args)

    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.to(args.device)
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Save losses and best epoch stats and model in dictionary
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0, 'best_model': model}
    test_dict = {"test_loss": [], 'jsd_test_copula': [], 't_1': [], 't_2': [], 'm_1': [], 'm_2': []}  # initialize a statistics dict

    # Train
    model, best_dict, test_dict = train_val(current_model=model,
                                            model_name='RealNVP',
                                            args=args,
                                            data_loaders=data_loaders,
                                            dataset=dataset,
                                            transform_inputs=False,
                                            cm_flow=False)

    output_copula = model.sample(num_samples=100000, transform=args.transform_fct)
    visualize_joint(output_copula.detach().numpy(), args, name='output_copula')

    dataset = datasets.distributions.Copula_Distr(args, transform=False)
    visualize_joint(dataset.trn.x, args, name='true_{}_copula_cm'.format(args.copula))

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   test_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # Plot pointwise copula difference
    jsd_graph(args,
              best_dict['best_validation_epoch'],
              model,
              cm_flow=True)
