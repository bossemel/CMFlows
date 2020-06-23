import torch
import torch.optim as optim
import torch.utils.data

import os
import numpy as np
from pathlib import Path
import random

from CM_modules.options import TrainOptions
import CM_modules.utils as utils
import CM_modules.flows as flows

from RealNVP import build_model as build_model_RealNVP
from RealNVP_modules.eval import jsd_graph
from DDSF import build_model as build_model_DDSF

from utils.visualizer import visualize_joint
from utils.save_statistics import save_statistics
from utils.various import HiddenPrints
import datasets.distributions

from experiment_runner import train_val

eps = 0.0001


def build_model(args):
    """Builds the CM Flow model. It is a concatenation of RealNVP and DDSF.

    Params:
        args: passed option arguments

    Returns:
        model: CM Flows model
        model_RealNVP: RealNVP model
        model_DDSF_1: 1st DDSF model
        model_DDSF_2: 2nd DDSF model
    """
    model_RealNVP = build_model_RealNVP(args)

    model_DDSF_1 = build_model_DDSF(args)
    model_DDSF_2 = build_model_DDSF(args)

    model = flows.CMFlow(transform=args.transform_fct,
                         model_RealNVP=model_RealNVP,
                         model_DDSF_1=model_DDSF_1,
                         model_DDSF_2=model_DDSF_2,
                         device=args.device,
                         batch_size=args.batch_size,
                         args=args)

    return model, model_RealNVP, model_DDSF_1, model_DDSF_2


def grid_search(args, model, gradient_clipping, num_inv_blocks,
                num_hidden_units, num_flow_layers, num_hidden_layers,
                num_ds_dims):
    results_dict = {}
    best_loss = 1000
    print('Grid search over: transform_functions, num_inv_blocks, num_hidden_units, weight_decay')
    for clip_grad_norm in gradient_clipping:
        args.clip_grad_norm = clip_grad_norm
        for num_blocks in num_inv_blocks:
            args.num_blocks = num_blocks
            for hidden_units in num_hidden_units:
                if num_hidden_layers > num_blocks:
                    args.hidden_units = hidden_units
                    for num_flow_layers_DDSF in num_flow_layers:
                        args.num_flow_layers_DDSF = num_flow_layers_DDSF
                        for hidden_layers in num_hidden_layers:
                            args.num_hid_layers_DDSF = hidden_layers
                            for ds_dims in num_ds_dims:
                                args.num_ds_dim = ds_dims
                                print(' clip_grad_norm:', clip_grad_norm,
                                      ' num_blocks:', num_blocks,
                                      ' num_hidden:', hidden_units,
                                      ' num_flow_layers_DDSF:', num_flow_layers_DDSF,
                                      ' num_hid_layers_DDSF: ', hidden_layers,
                                      ' num_ds_dim DDSF: ', ds_dims)
                                with HiddenPrints():
                                    current_model, current_best_dict, current_test_dict = train_and_plot(args=args,
                                                                                                         model=model,
                                                                                                         model_DDSF_1=model_DDSF_1,
                                                                                                         model_DDSF_2=model_DDSF_2,
                                                                                                         model_RealNVP=model_RealNVP,
                                                                                                         dataset=dataset,
                                                                                                         disable=True,
                                                                                                         grid_search=True)
                        current_hyperparams = (clip_grad_norm,
                                               num_blocks,
                                               hidden_units,
                                               num_flow_layers_DDSF,
                                               hidden_layers,
                                               ds_dims)
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
    with open(os.path.join(args.experiment_logs, 'grid_search.txt'), 'w') as f:
        f.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss) +
                'Best Epoch: ' + str(best_dict['best_validation_epoch']))
    return model, best_dict, test_dict


def train_and_plot(args, model, model_DDSF_1, model_DDSF_2, model_RealNVP, dataset, disable=False, grid_search=False):
    if args.pretrain_models:
        # Train
        args.optimizer = optim.Adam(model_DDSF_1.parameters(), lr=args.lr, betas=args.betas)
        model_DDSF_1, best_dict_DDSF_1, test_dict = train_val(current_model=model_DDSF_1,
                                                              model_name='DDSF_1',
                                                              args=args,
                                                              data_loaders=data_loaders,
                                                              dataset=dataset)
        args.optimizer = optim.Adam(model_DDSF_2.parameters(), lr=args.lr, betas=args.betas)
        model_DDSF_2, best_dict_DDSF_2, test_dict = train_val(current_model=model_DDSF_2,
                                                              model_name='DDSF_2',
                                                              args=args,
                                                              data_loaders=data_loaders,
                                                              dataset=dataset,
                                                              test_dict=test_dict)

        if not grid_search:
            # Visualize DDFS transformations
            vizdata = train_dataset

            n = vizdata.shape[0]
            context = torch.FloatTensor(n, 1).zero_().to(args.device)
            logdets = torch.FloatTensor(n).zero_().to(args.device)
            vizdata_1, __, __ = model_DDSF_1((vizdata[:, 0].reshape(-1, 1), logdets, context))
            vizdata_2, __, __ = model_DDSF_2((vizdata[:, 1].reshape(-1, 1), logdets, context))
            vizdata = torch.cat((vizdata_1, vizdata_2), dim=1)
            visualize_joint(vizdata.detach().numpy(), args, name='DDSF_output')

        args.optimizer = optim.Adam(model_RealNVP.parameters(), lr=args.lr, weight_decay=1e-6)

        model_RealNVP, best_dict_RealNVP, test_dict = train_val(model_RealNVP,
                                                                model_name='RealNVP',
                                                                args=args,
                                                                data_loaders=data_loaders,
                                                                dataset=dataset,
                                                                transform_model_1=model_DDSF_1,
                                                                transform_model_2=model_DDSF_1,
                                                                test_dict=test_dict)
        best_dict = best_dict_RealNVP

        if not grid_search:
            output_copula = model_RealNVP.sample_copula(num_samples=100000)
            visualize_joint(output_copula.detach().numpy(), args, name='output_copula_RealNVP')

            dataset = datasets.distributions.Copula_Distr(args, transform=False)
            visualize_joint(dataset.trn.x, args, name='true_{}_copula_cm'.format(args.copula))

            # Gather test losses and save statistics
            test_losses = {key: [np.mean(value)] for key, value in
                           test_dict.items()}  # save test set metrics in dict format
            epochs = (best_dict_DDSF_1['best_validation_epoch'],
                      best_dict_DDSF_2['best_validation_epoch'],
                      best_dict_RealNVP['best_validation_epoch'])
            save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                            # save test set metrics on disk in .csv format
                            stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=epochs)

            # Plot pointwise copula difference
            jsd_graph(args,
                      best_dict['best_validation_epoch'],
                      model_RealNVP)

    # # Train
    if args.train_cm_flow:
        args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=args.betas)
        model, best_dict, test_dict = train_val(model,
                                                model_name='CM_Flow',
                                                args=args,
                                                data_loaders=data_loaders,
                                                dataset=dataset)

        if not grid_search:
            output_copula = model.sample_copula(num_samples=100000)
            visualize_joint(output_copula.detach().numpy(), args, name='output_copula_cm')

            dataset = datasets.distributions.Copula_Distr(args, transform=False)
            visualize_joint(dataset.trn.x, args, name='true_{}_copula_cm'.format(args.copula))

            output_copula = model.sample(num_samples=100000)
            visualize_joint(output_copula.detach().numpy(), args, name='output_copula_normal_cm')

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
    dataset, data_loaders, train_dataset = utils.load_data(args)
    visualize_joint(dataset.trn.x, args, name='input_dataset')

    # Build model and send to device
    model, model_RealNVP, model_DDSF_1, model_DDSF_2 = build_model(args)
    model.state = dict()
    model_RealNVP.state = dict()
    model_DDSF_1.state = dict()
    model_DDSF_2.state = dict()

    model.to(args.device)

    # args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)

    args.RealNVP_part_of_CM_Flow = True

    # Train
    if args.grid_search:
        # Hyperparamter options:
        # General:
        gradient_clipping = [True, False]

        # RealNVP:
        num_inv_blocks = [4, 8, 16]
        num_hidden_units = [32, 64, 128]

        # DDSF:
        num_flow_layers = [5, 10]
        num_hidden_layers = [1, 2]
        num_ds_dims = [8, 16]

        model, best_dict, test_dict = grid_search(args=args,
                                                  model=model,
                                                  gradient_clipping=gradient_clipping,
                                                  num_inv_blocks=num_inv_blocks,
                                                  num_hidden_units=num_hidden_units,
                                                  num_flow_layers=num_flow_layers,
                                                  num_hidden_layers=num_hidden_layers,
                                                  num_ds_dims=num_ds_dims)

    else:
        train_and_plot(args)
