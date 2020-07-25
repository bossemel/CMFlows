import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from pathlib import Path
import random
import torch.optim as optim

from DDSF_modules import nn_modules as nn_, flows
from DDSF_modules.utils import load_data
from DDSF_modules.options import TrainOptions
from DDSF_modules.flows import MAF

from experiment_runner import train_val
from utils.load_and_save import save_statistics, load_model
from utils import HiddenPrints


def build_model(args):
    """Builds the DDSF model.

    Params:
        args: passed option arguments

    Returns:
        model: DDSF model
    """
    args.dimh = args.batch_size
    args.act = nn.ELU()
    args.dim = 1
    args.betas = (args.beta1, args.beta2)

    sequels = [nn_.SequentialFlow(
        flows.IAF_DDSF(dim=args.dim,
                       hid_dim=args.dimh,
                       context_dim=1,
                       num_layers=args.num_hid_layers_DDSF + 1,
                       activation=args.act,
                       device=args.device,
                       fixed_order=True,
                       num_ds_dim=args.num_ds_dim,
                       num_ds_layers=args.num_ds_layers),
        flows.FlipFlow(1)) for i in range(args.num_flow_layers_DDSF)] + \
        [flows.LinearFlow(args.dim, 1), ]

    model = MAF(args, *sequels)
    return model


def random_search(args):
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    ii = 0
    while ii < 50:
        args.num_flow_layers_DDSF = 2**np.random.choice(range(5))
        args.num_hid_layers_DDSF = 2**np.random.choice(range(5))
        args.dimh_DDSF = 2**np.random.choice(range(10))
        args.num_ds_dim = 2**np.random.choice(range(10))
        args.num_ds_layers = 2**np.random.choice(range(5))

        current_hyperparams = (args.num_flow_layers_DDSF,
                               args.num_hid_layers_DDSF,
                               args.dimh_DDSF,
                               args.num_ds_dim,
                               args.num_ds_layers)
        if current_hyperparams not in tested_combinations:
            print('Num. Flow Layers: {}, Num. Hidden Layers: {}, Num. Hidden Units: {},\
                Num. Sigm. Units: {}, Num. Sigm. Layers: {}'.format(args.num_flow_layers_DDSF,
                                                                    args.num_hid_layers_DDSF,
                                                                    args.dimh_DDSF,
                                                                    args.num_ds_dim,
                                                                    args.num_ds_layers))
            try:
                with HiddenPrints():
                    __, current_best_dict, current_test_dict = train_and_plot(args,
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
            except:
                print('Error for {}'.format(current_hyperparams))
                ii += 1
    print('Random search complete for {}'.format(args.marginal))
    print('Best hyperparams: {}'.format(best_hyperparams))
    print('Lowest Val Loss: {}'.format(best_loss))
    print('Lowest Val Loss Epoch: {}'.format(best_dict['best_validation_epoch']))
    with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'a') as f:
        f.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                'Best Epoch: ' + str(best_dict['best_validation_epoch']))


def grid_search(args, dataset, data_loaders, flow_layers, hidden_layers, hidden_units,
                deep_sigm_dim, deep_sigm_layers):
    results_dict = {}
    best_loss = 1000
    print('Grid search over: transform_functions, num_inv_blocks, num_hidden_units, weight_decay')
    for num_flows_layers_DDSF in flow_layers:
        args.num_flows_layers_DDSF = num_flows_layers_DDSF
        for num_hid_layers_DDSF in hidden_layers:
            args.num_hid_layers_DDSF = num_hid_layers_DDSF
            for dimh_DDSF in hidden_units:
                args.dimh_DDSF = dimh_DDSF
                for num_ds_dim in deep_sigm_dim:
                    args.num_ds_dim = num_ds_dim
                    for num_ds_layers in deep_sigm_layers:
                        args.num_ds_layers = num_ds_layers
                        print(' num_flows_layers_DDSF:', num_flows_layers_DDSF,
                              ' num_hid_layers_DDSF:', num_hid_layers_DDSF,
                              ' dimh_DDSF:', dimh_DDSF,
                              ' num_ds_dim', num_ds_dim,
                              ' num_ds_layers: ', num_ds_layers)
                        try:
                            with HiddenPrints():
                                current_model, current_best_dict, current_test_dict = train_and_plot(args,
                                                                                                     dataset,
                                                                                                     data_loaders,
                                                                                                     disable_tqdm=True,
                                                                                                     grid_search=True)
                            current_hyperparams = (num_flows_layers_DDSF,
                                                   num_hid_layers_DDSF,
                                                   dimh_DDSF,
                                                   num_ds_dim,
                                                   num_ds_layers)
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
                        except:
                            print('Error for {}'.format(current_hyperparams))
    print('Grid search complete for ', args.marginal)
    print('Best hyperparams: ', best_hyperparams)
    print('Lowest Val Loss: ', best_loss)
    print('Lowest Val Loss Epoch', best_dict['best_validation_epoch'])
    with open(os.path.join(args.experiment_logs, 'grid_search.txt'), 'a') as f:
        f.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                'Best Epoch: ' + str(best_dict['best_validation_epoch']))
    return model, best_dict, test_dict


def train_and_plot(args, dataset, data_loaders, disable_tqdm=False, grid_search=False, rvine=False, save_name=None):
    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.to(args.device)

    # Set optimizer
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    # Train
    best_dict, test_dict = train_val(model=model,
                                     model_name='DDSF',
                                     args=args,
                                     data_loaders=data_loaders,
                                     dataset=dataset,
                                     disable_tqdm=disable_tqdm,
                                     grid_search=grid_search,
                                     rvine=rvine,
                                     save_name=save_name)
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

    # Set up data loader
    dataset, data_loaders = load_data(args)

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    if args.random_search:
        random_search(args)
    elif args.grid_search:
        # Hyperparameter options:
        flow_layers = [5, 10]
        hidden_layers = [1, 2]
        hidden_units = [64, 128]
        deep_sigm_dim = [8, 16]
        deep_sigm_layers = [1, 2]

        # Perform Grid Search
        model, best_dict, test_dict = grid_search(args,
                                                  dataset,
                                                  data_loaders,
                                                  flow_layers,
                                                  hidden_layers,
                                                  hidden_units,
                                                  deep_sigm_dim,
                                                  deep_sigm_layers)
    else:
        # Train model
        model, best_dict, test_dict = train_and_plot(args,
                                                     dataset,
                                                     data_loaders)

        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict['best_validation_epoch'])

        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False,
                        test_epoch=best_dict['best_validation_epoch'])
