import os
import numpy as np
import torch.utils.data
import torch.nn as nn
from pathlib import Path
import random
import torch.optim as optim
import json

from DDSF_modules import nn_modules as nn_, flows
from DDSF_modules.utils import load_data
from DDSF_modules.options import TrainOptions
from DDSF_modules.flows import MAF

from experiment_runner import train_val
from utils.load_and_save import save_statistics, load_model
from utils import HiddenPrints


def build_model(args_):
    """Builds the DDSF model.

    Params:
        args: passed option arguments

    Returns:
        model: DDSF model
    """
    args_.dimh = args_.dimh_DDSF
    args_.act = nn.ELU()
    args_.dim = 1

    sequels = [nn_.SequentialFlow(
        flows.IAF_DDSF(dim=args_.dim,
                       hid_dim=args_.dimh,
                       context_dim=1,
                       num_layers=args_.num_hid_layers_DDSF + 1,
                       activation=args_.act,
                       device=args_.device,
                       fixed_order=True,
                       num_ds_dim=args_.num_ds_dim,
                       num_ds_layers=args_.num_ds_layers),
        flows.FlipFlow(1)) for __ in range(args_.num_flow_layers_DDSF)] + \
        [flows.LinearFlow(args_.dim, 1), ]

    ddsf_model = MAF(args_, *sequels)
    return ddsf_model


def random_search(args_):
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    ii = 0
    while ii < 200:
        args_.epochs = 80
        args_.num_flow_layers_DDSF = np.random.choice(range(1, 8))
        args_.num_hid_layers_DDSF = np.random.choice(range(1, 8))
        args_.dimh_DDSF = 2**np.random.choice(range(1, 7))
        args_.num_ds_dim = 2**np.random.choice(range(1, 7))
        args_.num_ds_layers = np.random.choice(range(1, 8))
        args_.lr = 1 / 10**np.random.choice(range(2, 6))
        args_.weight_decay = 1 / 10**(np.random.choice(range(2, 15)))
        args_.clip_grad_norm = np.random.choice([True, False])
        args_.amsgrad = np.random.choice([True, False])
        args_.clip_m = np.random.choice(range(1, 6))

        current_hyperparams = (args_.num_flow_layers_DDSF,
                               args_.num_hid_layers_DDSF,
                               args_.dimh_DDSF,
                               args_.num_ds_dim,
                               args_.num_ds_layers,
                               args_.weight_decay,
                               args_.lr,
                               args_.clip_grad_norm,
                               args_.amsgrad,
                               args_.clip_m)
        if current_hyperparams not in tested_combinations:
            print('Num. Flow Layers: {}, Num. Hidden Layers: {}, Num. Hidden Units: {},\
                Num. Sigm. Units: {}, Num. Sigm. Layers: {},\
                Weight Decay: {}, Learning Rate: {}, Clipping: {}, amsgrad: {}'.format(args_.num_flow_layers_DDSF,
                                                                                       args_.num_hid_layers_DDSF,
                                                                                       args_.dimh_DDSF,
                                                                                       args_.num_ds_dim,
                                                                                       args_.num_ds_layers,
                                                                                       args_.weight_decay,
                                                                                       args_.lr,
                                                                                       args_.clip_grad_norm,
                                                                                       args_.amsgrad,
                                                                                       args_.clip_m))
            with HiddenPrints():
                __, current_best_dict, current_test_dict = train_and_plot(args_,
                                                                          data_loaders_=data_loaders,
                                                                          disable_tqdm=True,
                                                                          hp_search=True)
            results_dict[current_hyperparams] = (current_best_dict['best_validation_epoch'],
                                                 current_best_dict['best_validation_loss'])
            print(results_dict[current_hyperparams])
            with open(os.path.join(args_.experiment_logs, 'random_search.txt'), 'w') as ff:
                ff.write(str(results_dict))
            if current_best_dict['best_validation_loss'] < best_loss:
                best_loss = current_best_dict['best_validation_loss']
                best_hyperparams = current_hyperparams
                best_dict_ = current_best_dict
            tested_combinations.append(current_hyperparams)
            ii += 1

    print('Random search complete for {}'.format(args_.marginal))
    print('Best hyperparams: {}'.format(best_hyperparams))
    print('Lowest Val Loss: {}'.format(best_loss))
    print('Lowest Val Loss Epoch: {}'.format(best_dict_['best_validation_epoch']))
    with open(os.path.join(args.experiment_logs, 'random_search.txt'), 'a') as ff:
        ff.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                 'Best Epoch: ' + str(best_dict_['best_validation_epoch']))


def train_and_plot(args_, data_loaders_, disable_tqdm=False, hp_search=False, rvine=False, save_name=None):
    # Build model and send to device
    model_ = build_model(args_)
    model_.state = dict()
    model_.to(args_.device)

    # Set optimizer
    args_.optimizer = optim.Adam(model_.parameters(), lr=args_.lr,
                                 weight_decay=args_.weight_decay, amsgrad=args_.amsgrad)
    args_.scheduler = optim.lr_scheduler.CosineAnnealingLR(args_.optimizer, args_.epochs)

    # Train
    best_dict_, test_dict_ = train_val(model=model_,
                                       model_name='marg_flow',
                                       args=args_,
                                       data_loaders=data_loaders_,
                                       disable_tqdm=disable_tqdm,
                                       hp_search=hp_search,
                                       rvine=rvine,
                                       save_name=save_name)
    return model_, best_dict_, test_dict_


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

    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    args.conditional_copula = False

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set up data loader
    dataset, data_loaders = load_data(args)

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    if args.random_search:
        random_search(args)
    else:
        # Train model
        model, best_dict, test_dict = train_and_plot(args,
                                                     data_loaders)

        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict['best_validation_epoch'])

        # Gather test losses and save statistics
        test_losses = {key: [torch.mean(torch.tensor(value))] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False,
                        test_epoch=best_dict['best_validation_epoch'])
