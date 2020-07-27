import torch
import torch.optim as optim
import torch.utils.data

import os
import numpy as np
from pathlib import Path
import random
import csv

from CM_modules.options import TrainOptions
import CM_modules.utils as utils
import CM_modules.flows as flows

from utils.visualizer import visualize_joint
from utils.load_and_save import save_statistics, load_statistics, load_model
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
    model = flows.CMFlow(transform=args.transform_fct,
                         device=args.device,
                         batch_size=args.batch_size,
                         args=args)

    return model


def visualize_DDSF_output(model, dataset, args):
    with torch.no_grad():
        vizdata = torch.tensor(dataset.trn)
        n = vizdata.shape[0]
        context = torch.FloatTensor(n, 1).zero_().to(args.device)
        logdets = torch.FloatTensor(n).zero_().to(args.device)
        vizdata_1, __, __ = model.model_DDSF_1.forward((vizdata[:, 0:1], logdets, context))
        vizdata_2, __, __ = model.model_DDSF_2.forward((vizdata[:, 1:2], logdets, context))
        vizdata = torch.cat((vizdata_1, vizdata_2), dim=1)
        if args.cuda:
            visualize_joint(vizdata, args, name='DDSF_output')
        else:
            visualize_joint(vizdata, args, name='DDSF_output')


def visualize_RealNVP_output(model, dataset, args):
    with torch.no_grad():
        # Visualize RealNVP outputs
        output_copula = model.sample_copula(num_samples=100000)
        visualize_joint(output_copula, args, name='output_copula_RealNVP')

        # Visualize true copula
        dataset = datasets.distributions.Copula_Distr(args, transform=False)
        visualize_joint(dataset.trn, args, name='true_{}_copula_cm'.format(args.copula))


def visualize_CM_Flow_output(model, dataset, args):
    with torch.no_grad():
        # Sample from the predicted copula
        output_copula = model.sample_copula(num_samples=100000)
        # Visualize the predicted copula
        if args.cuda:
            visualize_joint(output_copula, args, name='output_copula_cm')
        else:
            visualize_joint(output_copula, args, name='output_copula_cm')

        # Sample from the true copula and visualize it
        dataset = datasets.distributions.Copula_Distr(args, transform=False)
        visualize_joint(dataset.trn, args, name='true_{}_copula_cm'.format(args.copula))

        # Sample from the non-transformed copula (normal margins)
        output_copula = model.sample(num_samples=100000)
        if args.cuda:
            visualize_joint(output_copula, args, name='output_copula_normal_cm')
        else:
            visualize_joint(output_copula, args, name='output_copula_normal_cm')


def train_and_plot(args, dataset, data_loaders, disable_tqdm=False, error_bars=False, rvine=False):
    """Trains the CM Flow, saves test set results and plots.

    Params:
        args: passed arguments
        dataset: dataset class
        data_loaders: torch dataset loaders
        disable_tqdm: indicate whether to print progress bar
        error_bars: disables plotting and test set results for error bar calculation
        rvine: disables plotting and test set results for r-vine estimation
    """
    if not error_bars:
        visualize_joint(dataset.trn, args, name='input_dataset')

    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.to(args.device)

    # Pretrain models individually, with RealNVP using the outputs of DDSF as inputs
    if args.pretrain_models:
        # Train DDSFs
        args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        best_dict_DDSF_1, test_dict = train_val(model=model,
                                                model_name='DDSF_1',
                                                args=args,
                                                data_loaders=data_loaders,
                                                dataset=dataset,
                                                transform_inputs=True,
                                                disable_tqdm=disable_tqdm,
                                                error_bars=error_bars,
                                                rvine=rvine)

        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict_DDSF_1['best_validation_epoch'])

        args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        best_dict_DDSF_2, test_dict = train_val(model=model,
                                                model_name='DDSF_2',
                                                args=args,
                                                data_loaders=data_loaders,
                                                dataset=dataset,
                                                test_dict=test_dict,
                                                transform_inputs=True,
                                                disable_tqdm=disable_tqdm,
                                                error_bars=error_bars,
                                                rvine=rvine)

        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict_DDSF_2['best_validation_epoch'])

        # Visualize DDFS transformations
        if not error_bars and not rvine:
            visualize_DDSF_output(model, dataset, args)

        # Train RealNVP
        args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        best_dict_RealNVP, test_dict = train_val(model,
                                                 model_name='RealNVP',
                                                 args=args,
                                                 data_loaders=data_loaders,
                                                 dataset=dataset,
                                                 test_dict=test_dict,
                                                 transform_inputs=True,
                                                 disable_tqdm=disable_tqdm,
                                                 error_bars=error_bars,
                                                 rvine=rvine)

        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict_RealNVP['best_validation_epoch'])

        best_dict = best_dict_RealNVP

        if not error_bars and not rvine:
            visualize_RealNVP_output(model, dataset, args)

        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        sep = '_'
        epochs = sep.join(list([str(best_dict_DDSF_1['best_validation_epoch']),
                                str(best_dict_DDSF_2['best_validation_epoch']),
                                str(best_dict_RealNVP['best_validation_epoch'])]))
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=error_bars, test_epoch=epochs)

    # Train the CM Flow
    if args.train_cm_flow:

        # Set optimizer
        args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

        # Train model, and perform validation and test
        best_dict, test_dict = train_val(model,
                                         model_name='CM_Flow',
                                         args=args,
                                         data_loaders=data_loaders,
                                         dataset=dataset,
                                         disable_tqdm=disable_tqdm,
                                         error_bars=error_bars)

        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict['best_validation_epoch'])

        if not error_bars and not rvine:
            visualize_CM_Flow_output(model, dataset, args)

        # Gather test losses and save statistics
        test_losses = {key: [np.mean(value)] for key, value in
                       test_dict.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=error_bars, test_epoch=best_dict['best_validation_epoch'])
    return best_dict


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

    # Specify, that this RealNVP is part of a CM_Flow
    args.RealNVP_part_of_CM_Flow = True

    # Set up data loader
    dataset, data_loaders, train_dataset = utils.load_data(args)

    # Train model with specified options
    if args.error_bars is True:
        eval_dict = {}
        train_and_plot(args=args,
                       dataset=dataset,
                       data_loaders=data_loaders,
                       disable_tqdm=False,
                       error_bars=False)
        for ii in range(1, 10):
            train_and_plot(args=args,
                           dataset=dataset,
                           data_loaders=data_loaders,
                           disable_tqdm=False,
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
        train_and_plot(args=args,
                       dataset=dataset,
                       data_loaders=data_loaders,
                       disable_tqdm=False)
