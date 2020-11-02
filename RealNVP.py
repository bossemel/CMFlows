import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import os
import numpy as np
from pathlib import Path
import random
import csv

import RealNVP_modules.flows as fnn
import RealNVP_modules.utils as utils
from RealNVP_modules.options import TrainOptions

from utils.visualizer import visualize_joint
from utils import HiddenPrints
import datasets.distributions
from utils.load_and_save import save_statistics, load_statistics, load_model

from experiment_runner import train_val
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})


def build_model(args):
    """Creates model.

    Params:
        args: option input arguments

    Returns: RealNVP model
    """
    num_hidden = {args.copula: args.num_hidden_RealNVP}[args.copula]
    modules = []
    if args.conditional_copula:
        num_inputs = 1
        num_cond_inputs = 1
    else:
        num_inputs = 2
        num_cond_inputs = 0

    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(args.device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
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


def random_search(args):
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    ii = 0
    while ii < 30:
        args.transform_fct = np.random.choice(['gaussian', 'sigmoid'])
        pot_num_blocks = np.random.choice(range(7))
        args.num_blocks = 2**pot_num_blocks
        args.num_hidden = 2**np.random.choice(range(pot_num_blocks, 10))
        current_hyperparams = (args.transform_fct, args.num_blocks, args.num_hidden)
        if current_hyperparams not in tested_combinations:
            print('Transformation function: {}, Num. Invertible Blocks: {}, Num. Hidden Units: {}'.format(
                args.transform_fct, args.num_blocks, args.num_hidden))
            with HiddenPrints():
                __, current_best_dict, current_test_dict = train_and_plot(args,
                                                                          dataset=dataset,
                                                                          data_loaders=data_loaders,
                                                                          disable_tqdm=True,
                                                                          grid_search=True)
            results_dict[current_hyperparams] = (current_best_dict['best_validation_epoch'],
                                                 current_best_dict['best_validation_loss'])
            print(results_dict[current_hyperparams])
            with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'w') as f:
                f.write(str(results_dict))
            if current_best_dict['best_validation_loss'] < best_loss:
                best_loss = current_best_dict['best_validation_loss']
                best_hyperparams = current_hyperparams
                best_dict = current_best_dict
            tested_combinations.append(current_hyperparams)
            ii += 1
    print('Random search complete for {}'.format(args.copula))
    print('Best hyperparams: {}'.format(best_hyperparams))
    print('Lowest Val Loss: {}'.format(best_loss))
    print('Lowest Val Loss Epoch: {}'.format(best_dict['best_validation_epoch']))
    with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'a') as f:
        f.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                'Best Epoch: ' + str(best_dict['best_validation_epoch']))


def grid_search(args, dataset, data_loaders, transform_functions, num_inv_blocks, num_hidden_units):
    results_dict = {}
    best_loss = 1000
    print('Grid search over: transform_functions, num_inv_blocks, num_hidden_units')
    for transform_fct in transform_functions:
        args.transform_fct = transform_fct
        for num_blocks in num_inv_blocks:
            args.num_blocks = num_blocks
            for num_hidden in num_hidden_units:
                if num_hidden > num_blocks:
                    args.num_hidden = num_hidden
                    print(' transform_fct:', transform_fct,
                          ' num_blocks:', num_blocks,
                          ' num_hidden:', num_hidden)
                    with HiddenPrints():
                        current_model, current_best_dict, current_test_dict = train_and_plot(args,
                                                                                             dataset=dataset,
                                                                                             data_loaders=data_loaders,
                                                                                             disable_tqdm=True,
                                                                                             grid_search=True)
                    current_hyperparams = (transform_fct, num_blocks, num_hidden)
                    results_dict[current_hyperparams] = (current_best_dict['best_validation_epoch'],
                                                         current_best_dict['best_validation_loss'])
                    print(results_dict[current_hyperparams])
                    with open(os.path.join(args.experiment_logs, 'grid_search.txt'), 'w') as f:
                        f.write(str(results_dict))
                    if current_best_dict['best_validation_loss'] < best_loss:
                        best_loss = current_best_dict['best_validation_loss']
                        best_hyperparams = current_hyperparams
                        model = current_model
                        best_dict = current_best_dict
                        test_dict = current_test_dict
    print('Grid search complete for ', args.copula)
    print('Best hyperparams: ', best_hyperparams)
    print('Lowest Val Loss: ', best_loss)
    print('Lowest Val Loss Epoch', best_dict['best_validation_epoch'])
    with open(os.path.join(args.experiment_logs, 'grid_search.txt'), 'a') as f:
        f.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                'Best Epoch: ' + str(best_dict['best_validation_epoch']))
    return model, best_dict, test_dict


def train_and_plot(args, dataset, data_loaders, disable_tqdm=False, grid_search=False, rvine=False, error_bars=False, save_name=None):
    if not grid_search and not rvine and not error_bars:
        visualize_joint(dataset.trn, args.figures_path, name='input_dataset')

    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.to(args.device)

    # Set optimizer
    #args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.scheduler = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, args.epochs) #, args.num_training_steps, 0)

    # Train
    best_dict, test_dict = train_val(model=model,
                                     model_name='RealNVP',
                                     args=args,
                                     data_loaders=data_loaders,
                                     dataset=dataset,
                                     transform_inputs=False,
                                     disable_tqdm=disable_tqdm,
                                     error_bars=error_bars,
                                     grid_search=grid_search,
                                     rvine=rvine,
                                     save_name=save_name)

    if not rvine:
        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=error_bars, test_epoch=best_dict['best_validation_epoch'])

    return model, best_dict, test_dict


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

    # Specify, that this RealNVP is not part of a CM_Flow
    args.RealNVP_part_of_CM_Flow = False

    # Set up data loader
    dataset, data_loaders = utils.load_data(args)

    if args.random_search:
        random_search(args=args)
    elif args.grid_search:
        # Hyperparameter options:
        transform_functions = ['gaussian', 'sigmoid']
        num_inv_blocks = [4, 8, 16]
        num_hidden_units = [32, 64, 128]
        grid_search(args=args,
                    dataset=dataset,
                    data_loaders=data_loaders,
                    transform_functions=transform_functions,
                    num_inv_blocks=num_inv_blocks,
                    num_hidden_units=num_hidden_units)
    else:
        if args.error_bars is True:
            eval_dict = {}
            # Train model
            model, best_dict, test_dict = train_and_plot(args=args,
                                                         dataset=dataset,
                                                         data_loaders=data_loaders,
                                                         disable_tqdm=True,
                                                         grid_search=False,
                                                         error_bars=False)

            model = load_model(model, args.experiment_saved_models, 'train_model',
                               best_dict['best_validation_epoch'])
            for ii in range(1, 10):
                train_and_plot(args=args,
                               dataset=dataset,
                               data_loaders=data_loaders,
                               disable_tqdm=True,
                               error_bars=True)
            stats_dict = load_statistics(args.experiment_logs, 'test_summary.csv')
            with open(os.path.join(args.experiment_logs, 'error_bars.csv'), 'w') as f:
                writer = csv.writer(f)
                for key in stats_dict.keys():
                    if key != 'epoch':
                        float_list = np.array([float(xx) for xx in stats_dict[key]])
                        line = [key, np.mean(float_list), np.std(float_list)]
                        writer.writerow(line)
        else:
            # Train model
            model, best_dict, test_dict = train_and_plot(args=args,
                                                         dataset=dataset,
                                                         data_loaders=data_loaders,
                                                         disable_tqdm=False,
                                                         grid_search=False)

            model = load_model(model, args.experiment_saved_models, 'train_model',
                               best_dict['best_validation_epoch'])
            # Sample from predicted copual and visualize it
            with torch.no_grad():
                if args.conditional_copula:
                    cond_inputs = torch.tensor(np.random.normal(size=(100000, 1))).float()
                    output_copula = model.sample(num_samples=100000, cond_inputs=cond_inputs, transform=args.transform_fct, device=args.device).cpu()
                    visualize_joint(output_copula, args.figures_path, name='output_copula')
                    output_copula = model.sample(num_samples=100000, cond_inputs=cond_inputs, transform=None, device=args.device).cpu()
                    visualize_joint(output_copula, args.figures_path, name='output_copula_untransformed')
                else:
                    output_copula = model.sample(num_samples=100000, transform=args.transform_fct, device=args.device).cpu()
                    visualize_joint(output_copula, args.figures_path, name='output_copula')
                    output_copula = model.sample(num_samples=100000, transform=None, device=args.device).cpu()
                    visualize_joint(output_copula, args.figures_path, name='output_copula_untransformed')

        # Sample from true copula and visualize it
        obs = args.obs
        args.obs = 100000
        dataset = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs, transform=False)
        visualize_joint(dataset.trn, args.figures_path, name='true_{}_copula_cm'.format(args.copula))
        args.obs = obs

        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])
