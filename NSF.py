import torch
import torch.optim as optim
import torch.utils.data
import os
import numpy as np
import random
import csv
import matplotlib
import json

import DDSF_modules.utils as ddsf_utils

from utils.visualizer import visualize_joint
import datasets.distributions
from utils.load_and_save import save_statistics, load_statistics, load_model

from experiment_runner import train_val

from NSF_modules import flows
from NSF_modules.options import TrainOptions
from NSF_modules.load_data import load_data as load_cop_data
from utils import HiddenPrints, create_folders

matplotlib.rcParams.update({'figure.max_open_warning': 0})


def build_model(args_, flow_type='cop_flow'):
    """Creates model.

    Params:
        args_: option input arguments

    Returns: RealNVP model
    """
    if flow_type == 'cop_flow':
        if args_.conditional_copula:
            num_inputs = 1
            context_ = 1
        else:
            num_inputs = 2
            context_ = 0
    elif flow_type == 'marg_flow':
        num_inputs = 1
        context_ = 0
    else:
        raise ValueError('Unknown flow type')
    flow = flows.ConditionalFlow(dim=num_inputs,
                                 context_dim=context_, args=args_)
    return flow


def random_search(args_):
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    jj = args_.continue_from
    np.random.seed(jj)
    random.seed(jj)

    while jj < 200:
        args_.epochs = 100
        n_layers = np.random.choice(range(1, 10))
        hidden_units = 2**np.random.choice(range(1, 7))
        n_blocks = np.random.choice(range(1, 10))
        n_bins = 5 * np.random.choice(range(2, 10))
        lr = 1 / 10**np.random.choice(range(2, 5))
        weight_decay = 1 / 10**(np.random.choice(range(2, 15)))
        tail_bound = 2**np.random.choice(range(5, 8)).item()
        amsgrad = np.random.choice([True, False])
        clip_grad_norm = np.random.choice([True, False])
        identity_init = np.random.choice([True, False])
        tails = np.random.choice(['linear', None])

        if args_.flow_type == 'cop_flow':
            args_.n_layers_c = n_layers
            args_.hidden_units_c = hidden_units
            args_.n_blocks_c = n_blocks
            args_.n_bins_c = n_bins
            args_.dropout_c = 0.05 * np.random.choice(range(1, 6))
            args_.lr_c = lr
            args_.weight_decay_c = weight_decay
            args_.tail_bound_c = tail_bound
            args_.use_batch_norm_c = np.random.choice([True, False])
            args_.amsgrad_c = amsgrad
            args_.clip_grad_norm = clip_grad_norm
            args_.identity_init_c = identity_init
            args_.tails_c = tails
            current_hyperparams = (args_.n_layers_c,
                                   args_.hidden_units_c,
                                   args_.n_blocks_c,
                                   args_.n_bins_c,
                                   args_.dropout_c,
                                   args_.lr_c,
                                   args_.weight_decay_c,
                                   args_.tail_bound_c,
                                   args_.use_batch_norm_c,
                                   args_.amsgrad_c,
                                   args_.clip_grad_norm,
                                   args_.identity_init_c,
                                   args_.tails_c)

        elif args_.flow_type == 'marg_flow':
            args_.n_layers_m = n_layers
            args_.hidden_units_m = hidden_units
            args_.n_blocks_m = n_blocks
            args_.n_bins_m = n_bins
            args_.lr_m = lr
            args_.weight_decay_m = weight_decay
            args_.tail_bound_m = tail_bound
            args_.amsgrad_m = amsgrad
            args_.clip_grad_norm = clip_grad_norm
            args_.identity_init_m = identity_init
            args_.tails_m = tails
            current_hyperparams = (args_.n_layers_m,
                                   args_.hidden_units_m,
                                   args_.n_blocks_m,
                                   args_.n_bins_m,
                                   args_.lr_m,
                                   args_.weight_decay_m,
                                   args_.clip_grad_norm,
                                   args_.tail_bound_m,
                                   args_.identity_init_m,
                                   args_.tails_m)
        else:
            raise ValueError('Unknown Flow type')

        if current_hyperparams not in tested_combinations:
            if args_.flow_type == 'cop_flow':
                hyperparams_string = 'n_layers, hidden_units, n_blocks, n_bins, dropout, lr, weight_decay, \
                tail_bound, batch_norm, amsgrad, clip_grad'
            else:
                hyperparams_string = 'n_layers, hidden_units, n_blocks, n_bins, lr, weight_decay, clip_grad_norm, \
                tail_bound, identity_init_m, tails_m'
            print('{}: {}'.format(hyperparams_string, current_hyperparams))
            with HiddenPrints():
                __, current_best_dict, current_test_dict = train_and_plot(args_,
                                                                          dataset_=dataset,
                                                                          data_loaders_=data_loaders,
                                                                          disable_tqdm=True,
                                                                          hp_search=True)
            results_dict[current_hyperparams] = (current_best_dict['best_validation_epoch'],
                                                 current_best_dict['best_validation_loss'])
            print(results_dict[current_hyperparams])
            with open(os.path.join(args_.experiment_logs, 'random_search.txt'), 'w' if jj == 0 else 'a') as ff:
                if jj == 0:
                    ff.write(hyperparams_string + '\n' + str(current_hyperparams) + ': ' +
                             str(results_dict[current_hyperparams]) + '\n')
                else:
                    ff.write(str(current_hyperparams) + ': ' + str(results_dict[current_hyperparams]) + '\n')

            if current_best_dict['best_validation_loss'] < best_loss:
                best_loss = current_best_dict['best_validation_loss']
                best_hyperparams = current_hyperparams
                best_dict_ = current_best_dict
            tested_combinations.append(current_hyperparams)
            jj += 1
    if args_.continue_from == 0:
        print('Random search complete for {}'.format(args_.copula if args_.flow_type == 'cop_flow' else args_.marginal))
        print('Best hyperparams: {}'.format(best_hyperparams))
        print('Lowest Val Loss: {}'.format(best_loss))
        print('Lowest Val Loss Epoch: {}'.format(best_dict_['best_validation_epoch']))
        if args.continue_from == 0:
            with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'a') as ff:
                ff.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                         'Best Epoch: ' + str(best_dict_['best_validation_epoch']))
    else:
        print('Random search complete. See random_search.txt for results.')


def train_and_plot(args_, dataset_, data_loaders_, disable_tqdm=False, hp_search=False, rvine=False,
                   error_bars=False, save_name=None):
    if not hp_search and not rvine and not error_bars and args_.flow_type == 'cop_flow':
        visualize_joint(dataset_.trn, args_.figures_path, name='input_dataset')

    # Build model and send to device
    model_ = build_model(args_, flow_type=args_.flow_type)
    model_.state = dict()
    model_.to(args_.device)
    model_.train()

    # Set optimizer
    args_.optimizer = optim.Adam(model_.parameters(), lr=args_.lr_c if args_.flow_type == 'cop_flow' else args_.lr_m,
                                 weight_decay=args_.weight_decay_c if args_.flow_type == 'cop_flow'
                                 else args_.weight_decay_m,
                                 amsgrad=args_.amsgrad_c if args_.flow_type == 'cop_flow' else args_.amsgrad_m)
    args_.scheduler = optim.lr_scheduler.CosineAnnealingLR(args_.optimizer, args_.epochs)

    # Train
    best_dict_, test_dict_ = train_val(model=model_,
                                       model_name=args_.flow_type,
                                       args=args_,
                                       data_loaders=data_loaders_,
                                       disable_tqdm=disable_tqdm,
                                       error_bars=error_bars,
                                       hp_search=hp_search,
                                       rvine=rvine,
                                       save_name=save_name)

    if not rvine:
        # Gather test losses and save statistics
        test_losses_ = {kk: [torch.mean(torch.tensor(value))] for kk, value in
                        test_dict_.items()}
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        stats_dict=test_losses_, current_epoch=0, continue_from_mode=error_bars,
                        test_epoch=best_dict_['best_validation_epoch'])

    return model_, best_dict_, test_dict_


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    create_folders(args)
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
        dataset, data_loaders = load_cop_data(args)
    elif args.flow_type == 'marg_flow':
        dataset, data_loaders = ddsf_utils.load_data(args)

    if args.random_search:
        random_search(args_=args)
    else:
        if args.error_bars is True:
            eval_dict = {}
            # Train model
            model, best_dict, test_dict = train_and_plot(args_=args,
                                                         dataset_=dataset,
                                                         data_loaders_=data_loaders,
                                                         disable_tqdm=True,
                                                         hp_search=False,
                                                         error_bars=False)

            model = load_model(model, args.experiment_saved_models, 'train_model',
                               best_dict['best_validation_epoch'])
            for ii in range(1, 10):
                train_and_plot(args_=args,
                               dataset_=dataset,
                               data_loaders_=data_loaders,
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
            model, best_dict, test_dict = train_and_plot(args_=args,
                                                         dataset_=dataset,
                                                         data_loaders_=data_loaders,
                                                         disable_tqdm=False,
                                                         hp_search=False)

            model = load_model(model, args.experiment_saved_models, 'train_model',
                               best_dict['best_validation_epoch']).to(args.device)
            # Sample from predicted copual and visualize it
            with torch.no_grad():
                # if args.flow_type == 'marg_flow':
                #     norm = torch.distributions.normal.Normal(loc=0, scale=1)
                #     marg_flow_noise = model.flow.transform_to_noise(torch.tensor(dataset.trn).to(args.device)).detach().cpu().reshape(-1, 1)
                #     visualize_joint(norm.cdf(torch.cat([marg_flow_noise, marg_flow_noise], axis=1)), args.figures_path, name='outputs_marginal_noise')
                #     sample = model.flow.sample(num_samples=10000).reshape(-1, 1)
                #     visualize_joint(norm.cdf(torch.cat([sample, sample], axis=1)), args.figures_path, name='outputs_marginal_sample')

                if args.flow_type == 'cop_flow':
                    if args.conditional_copula:
                        context = torch.tensor(np.random.normal(size=(100000, 1))).float()
                        output_copula = model.sample_copula(num_samples=100000, context=context,
                                                            device=args.device).cpu()
                        visualize_joint(output_copula, args.figures_path, name='output_copula')
                        output_copula = model.sample(num_samples=100000, context=context, transform=None,
                                                     device=args.device).cpu()
                        visualize_joint(output_copula, args.figures_path, name='output_copula_untransformed')
                    else:
                        output_copula = model.sample_copula(num_samples=100000, device=args.device)
                        visualize_joint(output_copula.cpu(), args.figures_path, name='output_copula')
                        output_copula = model.sample(num_samples=100000, device=args.device).cpu()
                        visualize_joint(output_copula, args.figures_path, name='output_copula_untransformed')

                    # Sample from true copula and visualize it
                    obs = args.obs
                    args.obs = 100000
                    dataset = datasets.distributions.Copula_Distr(args.copula, args.theta, args.obs)
                    visualize_joint(dataset.trn, args.figures_path, name='true_{}_copula_cm'.format(args.copula))
                    args.obs = obs

            # Gather test losses and save statistics
            test_losses = {key: [torch.mean(torch.tensor(value))] for key, value in
                           test_dict.items()}  # save test set metrics in dict format
            save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                            # save test set metrics on disk in .csv format
                            stats_dict=test_losses, current_epoch=0, continue_from_mode=False,
                            test_epoch=best_dict['best_validation_epoch'])
