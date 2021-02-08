import torch
import torch.optim as optim
import torch.utils.data
import os
import numpy as np
from pathlib import Path
import random
import csv
import matplotlib
import json

import RealNVP_modules.utils as RealNVP_utils
import DDSF_modules.utils as DDSF_utils

from utils.visualizer import visualize_joint
import datasets.distributions
from utils.load_and_save import save_statistics, load_statistics, load_model

from experiment_runner import train_val

from NSF_modules import flows
from NSF_modules.options import TrainOptions
from utils import HiddenPrints

matplotlib.rcParams.update({'figure.max_open_warning': 0})


def build_model(args, flow_type='cop_flow'):
    """Creates model.

    Params:
        args: option input arguments

    Returns: RealNVP model
    """
    if flow_type == 'cop_flow':
        if args.conditional_copula:
            num_inputs = 1
            context = 1
        else:
            num_inputs = 2
            context = 0
    elif flow_type == 'marg_flow':
        num_inputs = 1
        context = 0
    else:
        raise ValueError('Unknown flow type')
    flow = flows.ConditionalFlow(dim=num_inputs,
                                 context_dim=context, args=args)
    return flow


def random_search(args):
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    ii = args.continue_from
    np.random.seed(ii)
    random.seed(ii)

    while ii < 200:
        args.epochs = 50
        n_layers = np.random.choice(range(1, 10))
        hidden_units = 2**np.random.choice(range(1, 8))
        n_blocks = np.random.choice(range(1, 10))
        n_bins = 5 * np.random.choice(range(2, 10))
        lr_number = np.random.choice(range(2, 5))
        lr = 1 / 10**lr_number
        weight_decay = 1 / 10**(np.random.choice(range(2, 15)))
        tail_bound = 2**np.random.choice(range(5, 7))

        if args.flow_type == 'cop_flow':
            args.n_layers_c = n_layers
            args.hidden_units_c = hidden_units
            args.n_blocks_c = n_blocks
            args.n_bins_c = n_bins
            args.dropout_c = 0.05 * np.random.choice(range(1, 6))
            args.lr_c = lr
            args.weight_decay_c = weight_decay
            args.tail_bound_c = tail_bound
            current_hyperparams = (args.n_layers_c,
                                   args.hidden_units_c,
                                   args.n_blocks_c,
                                   args.n_bins_c,
                                   args.dropout_c,
                                   args.lr_c,
                                   args.weight_decay_c,
                                   args.tail_bound_c)

        elif args.flow_type == 'marg_flow':
            args.n_layers_m = n_layers
            args.hidden_units_m = hidden_units
            args.n_blocks_m = n_blocks
            args.n_bins_m = n_bins
            args.lr_m = lr
            args.weight_decay_m = weight_decay
            args.tail_bound_m = tail_bound
            current_hyperparams = (args.n_layers_m,
                                   args.hidden_units_m,
                                   args.n_blocks_m,
                                   args.n_bins_m,
                                   args.lr_m,
                                   args.weight_decay_m,
                                   args.clip_grad_norm,
                                   args.tail_bound_m)
        else:
            raise ValueError('Unknown Flow type')

        if current_hyperparams not in tested_combinations:
            if args.flow_type == 'cop_flow':
                hyperparams_string = 'n_layers, hidden_units, n_blocks, n_bins, dropout, lr, weight_decay, tail_bound'
            else:
                hyperparams_string = 'n_layers, hidden_units, n_blocks, n_bins, lr, weight_decay, tail_bound'
            print('{}: {}'.format(hyperparams_string, current_hyperparams))
            with HiddenPrints():
                __, current_best_dict, current_test_dict = train_and_plot(args,
                                                                          dataset=dataset,
                                                                          data_loaders=data_loaders,
                                                                          disable_tqdm=True,
                                                                          hp_search=True)
            results_dict[current_hyperparams] = (current_best_dict['best_validation_epoch'],
                                                 current_best_dict['best_validation_loss'])
            print(results_dict[current_hyperparams])
            with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'w' if ii == 0 else 'a') as f:
                if ii == 0:
                    f.write(hyperparams_string + '\n' + str(current_hyperparams) + ': ' + str(results_dict[current_hyperparams]) + '\n')
                else:
                    f.write(str(current_hyperparams) + ': ' + str(results_dict[current_hyperparams]) + '\n')

            if current_best_dict['best_validation_loss'] < best_loss:
                best_loss = current_best_dict['best_validation_loss']
                best_hyperparams = current_hyperparams
                best_dict = current_best_dict
            tested_combinations.append(current_hyperparams)
            ii += 1
    if args.continue_from == 0:
        print('Random search complete for {}'.format(args.copula if args.flow_type == 'cop_flow' else args.marginal))
        print('Best hyperparams: {}'.format(best_hyperparams))
        print('Lowest Val Loss: {}'.format(best_loss))
        print('Lowest Val Loss Epoch: {}'.format(best_dict['best_validation_epoch']))
        if args.continue_from == 0:
            with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'a') as f:
                f.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                        'Best Epoch: ' + str(best_dict['best_validation_epoch']))
    else:
        print('Random search complete. See random_search.txt for results.')


def train_and_plot(args, dataset, data_loaders, disable_tqdm=False, hp_search=False, rvine=False, error_bars=False, save_name=None):
    if not hp_search and not rvine and not error_bars and args.flow_type == 'cop_flow':
        visualize_joint(dataset.trn, args.figures_path, name='input_dataset')

    # Build model and send to device
    model = build_model(args, flow_type=args.flow_type)
    model.state = dict()
    model.to(args.device)
    model.train()

    # Set optimizer
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr_c if args.flow_type == 'cop_flow' else args.lr_m,
                                weight_decay=args.weight_decay_c if args.flow_type == 'cop_flow' else args.weight_decay_m)
    args.scheduler = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, args.epochs) #, args.num_training_steps, 0)

    # Train
    best_dict, test_dict = train_val(model=model,
                                     model_name=args.flow_type,
                                     args=args,
                                     data_loaders=data_loaders,
                                     disable_tqdm=disable_tqdm,
                                     error_bars=error_bars,
                                     hp_search=hp_search,
                                     rvine=rvine,
                                     save_name=save_name)

    if not rvine:
        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=error_bars,
                        test_epoch=best_dict['best_validation_epoch'])

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
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

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
    args.cop_flow_part_of_CM_Flow = False

    # Set up data loader
    if args.flow_type == 'cop_flow':
        dataset, data_loaders = RealNVP_utils.load_data(args)
    elif args.flow_type == 'marg_flow':
        dataset, data_loaders = DDSF_utils.load_data(args)

    if args.random_search:
        random_search(args=args)
    else:
        if args.error_bars is True:
            eval_dict = {}
            # Train model
            model, best_dict, test_dict = train_and_plot(args=args,
                                                         dataset=dataset,
                                                         data_loaders=data_loaders,
                                                         disable_tqdm=True,
                                                         hp_search=False,
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
                                                         hp_search=False)

            model = load_model(model, args.experiment_saved_models, 'train_model',
                               best_dict['best_validation_epoch'])
            # Sample from predicted copual and visualize it
            with torch.no_grad():
                if args.flow_type == 'marg_flow':
                    norm = torch.distributions.normal.Normal(loc=0, scale=1)
                    marg_flow_noise = model.flow.transform_to_noise(torch.tensor(dataset.trn)).reshape(-1, 1)
                    visualize_joint(norm.cdf(torch.cat([marg_flow_noise, marg_flow_noise], axis=1)), args.figures_path, name='outputs_marginal_noise')
                    sample = model.flow.sample(num_samples=10000).reshape(-1, 1)
                    visualize_joint(norm.cdf(torch.cat([sample, sample], axis=1)), args.figures_path, name='outputs_marginal_sample')

                if args.flow_type == 'cop_flow':
                    if args.conditional_copula:
                        context = torch.tensor(np.random.normal(size=(100000, 1))).float()
                        output_copula = model.sample_copula(num_samples=100000, context=context, device=args.device).cpu()
                        visualize_joint(output_copula, args.figures_path, name='output_copula')
                        output_copula = model.sample(num_samples=100000, context=context, transform=None, device=args.device).cpu()
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

