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

from NSF_modules.visualizer import visualize1D

from utils.visualizer import visualize_joint
from utils.load_and_save import save_statistics, load_statistics, load_model
import datasets.distributions

from experiment_runner import train_val
import json

from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats
eps = 1e-07


def build_model(args):
    """Builds the CM Flow model. It is a concatenation of cop_flow and marg_flow.

    Params:
        args: passed option arguments

    Returns:
        model: CM Flows model
        cop_flow: cop_flow model
        marg_flow_1: 1st marg_flow model
        marg_flow_2: 2nd marg_flow model
    """
    model = flows.CMFlow(transform=args.transform_fct,
                         device=args.device,
                         batch_size=args.batch_size,
                         args=args)

    return model


def visualize_cop_flow_output(model, dataset, args):
    with torch.no_grad():
        if args.conditional_copula:
            context = torch.tensor(np.random.normal(size=(100000, 1))).float()
            output_copula = model.cop_flow.sample(num_samples=100000, context=context, transform=args.transform_fct, device=args.device).cpu()
            visualize_joint(output_copula, args.figures_path, name='output_copula')
            output_copula = model.cop_flow.sample(num_samples=100000, context=context, transform=None, device=args.device).cpu()
            visualize_joint(output_copula, args.figures_path, name='output_copula_untransformed')
        else:
            output_copula = model.cop_flow.sample(num_samples=100000, transform=args.transform_fct, device=args.device).cpu()
            visualize_joint(output_copula, args.figures_path, name='output_copula')
            output_copula = model.cop_flow.sample(num_samples=100000, transform=None, device=args.device).cpu()
            visualize_joint(output_copula, args.figures_path, name='output_copula_untransformed')


def visualize_CM_Flow_output(model, dataset, args):
    with torch.no_grad():
        # Sample from the predicted copula
        output_copula = model.sample_copula(num_samples=100000).cpu()
        # Visualize the predicted copula
        if args.cuda:
            visualize_joint(output_copula, args.figures_path, name='output_copula_cm')
        else:
            visualize_joint(output_copula, args.figures_path, name='output_copula_cm')

        # Sample from the true copula and visualize it
        dataset = datasets.distributions.Copula_Distr(args.copula, args.theta, obs=args.obs, transform=False)
        visualize_joint(dataset.trn, args.figures_path, name='true_{}_copula_cm'.format(args.copula))

        # Sample from the non-transformed copula (normal margins)
        output_copula = model.sample(num_samples=100000).cpu()
        if args.cuda:
            visualize_joint(output_copula, args.figures_path, name='output_copula_normal_cm')
        else:
            visualize_joint(output_copula, args.figures_path, name='output_copula_normal_cm')


def train_marginals(model, disable_tqdm, error_bars, rvine):
    # Pretrain models individually, with cop_flow using the outputs of marg_flow as inputs
    # Train marg_flows
    args.epochs = args.epochs_m
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr_m, weight_decay=args.weight_decay_m)

    # args.optimizer = optim.Adam(model.marg_flow_1.parameters(), lr=args.lr_m, weight_decay=args.weight_decay_m)
    # args.scheduler = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, args.epochs) #, args.num_training_steps, 0)

    for param in model.marg_flow_2.parameters():
        param.requires_grad = False
    for param in model.cop_flow.parameters():
        param.requires_grad = False

    args.marginal = args.marginal_1
    best_dict_marg_flow_1, test_dict = train_val(model=model.marg_flow_1,
                                                 model_name='marg_flow_1',
                                                 args=args,
                                                 data_loaders=data_loaders,
                                                 disable_tqdm=disable_tqdm,
                                                 error_bars=error_bars,
                                                 rvine=rvine,
                                                 cm_flow=True)

    model.marg_flow_1 = load_model(model.marg_flow_1, args.experiment_saved_models, 'best_epoch_model',
                                   best_dict_marg_flow_1['best_validation_epoch'])

    model.marg_flow_1.eval()

    visualize1D(model=model.marg_flow_1,
                epoch=best_dict_marg_flow_1['best_validation_epoch'],
                args=args,
                best_val=True,
                name='marg_flow_1')

    if args.marg_flow == 'NSF':
        marg_flow_1_output = model.marg_flow_1.flow.transform_to_noise(train_dataset[:, 0:1].to(args.device)).reshape(-1, 1)
    elif args.marg_flow == 'DDSF':
        marg_flow_1_output = model.marg_flow_1.transform_to_noise(train_dataset[:, 0:1].to(args.device)).reshape(-1, 1)

    args.optimizer = optim.Adam(model.parameters(), lr=args.lr_m, weight_decay=args.weight_decay_m)
    # args.optimizer = optim.Adam(model.marg_flow_2.parameters(), lr=args.lr_m, weight_decay=args.weight_decay_m)
    # args.scheduler = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, args.epochs)

    for param in model.marg_flow_1.parameters():
        param.requires_grad = False
    for param in model.marg_flow_2.parameters():
        param.requires_grad = True
    for param in model.cop_flow.parameters():
        param.requires_grad = False

    args.marginal = args.marginal_1

    best_dict_marg_flow_2, test_dict = train_val(model=model.marg_flow_2,
                                                 model_name='marg_flow_2',
                                                 args=args,
                                                 data_loaders=data_loaders,
                                                 disable_tqdm=disable_tqdm,
                                                 error_bars=error_bars,
                                                 rvine=rvine,
                                                 cm_flow=True)

    model.marg_flow_2 = load_model(model.marg_flow_2, args.experiment_saved_models, 'best_epoch_model',
                                   best_dict_marg_flow_2['best_validation_epoch'])

    model.marg_flow_2.eval()

    visualize1D(model=model.marg_flow_2,
                epoch=best_dict_marg_flow_2['best_validation_epoch'],
                args=args,
                best_val=True,
                name='marg_flow_2')

    for param in model.marg_flow_2.parameters():
        param.requires_grad = False

    if args.marg_flow == 'NSF':
        marg_flow_2_output = model.marg_flow_2.flow.transform_to_noise(train_dataset[:, 1:2].to(args.device)).reshape(-1, 1)
    elif args.marg_flow == 'DDSF':
        marg_flow_2_output = model.marg_flow_2.transform_to_noise(train_dataset[:, 1:2].to(args.device)).reshape(-1, 1)

    return model, best_dict_marg_flow_1, best_dict_marg_flow_2, marg_flow_1_output, marg_flow_2_output


def transform_dataset(model, train_dataset, marg_flow_1_output, marg_flow_2_output):
    with torch.no_grad():
        train_dataset = torch.cat((marg_flow_1_output, marg_flow_2_output), dim=1).cpu()
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)

        data_loaders['train_loader'] = train_loader
        dataset.trn = train_dataset

        normal_distr = torch.distributions.normal.Normal(0, 1)
        train_dataset_uniform = normal_distr.cdf(train_dataset)
        visualize_joint(train_dataset, args.figures_path, name='marg_flow_transform_output')
        visualize_joint(train_dataset_uniform, args.figures_path, name='marg_flow_transform_output_uniform')
    return data_loaders, dataset.trn


def train_copula_flow(model, train_dataset, disable_tqdm, error_bars, rvine, marg_flow_1_output, marg_flow_2_output):
    # Train cop_flow
    args.epochs = args.epochs_c
    args.optimizer = optim.Adam(model.cop_flow.parameters(), lr=args.lr_c, weight_decay=args.weight_decay_c)
    args.scheduler = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, args.epochs)

    data_loaders, dataset.trn = transform_dataset(model, train_dataset, marg_flow_1_output, marg_flow_2_output)

    best_dict_cop_flow, test_dict = train_val(model.cop_flow,
                                              model_name='cop_flow',
                                              args=args,
                                              data_loaders=data_loaders,
                                              disable_tqdm=disable_tqdm,
                                              error_bars=error_bars,
                                              rvine=rvine,
                                              cm_flow=True)

    model.cop_flow = load_model(model.cop_flow, args.experiment_saved_models, 'best_epoch_model',
                                best_dict_cop_flow['best_validation_epoch'])

    model.cop_flow.eval()

    best_dict = best_dict_cop_flow

    if not error_bars and not rvine:
        visualize_cop_flow_output(model, dataset, args)
    return model, best_dict, test_dict, best_dict_cop_flow


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
        visualize_joint(dataset.trn, args.figures_path, name='input_dataset')

    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.marg_flow_1.state = dict()
    model.marg_flow_2.state = dict()
    model.cop_flow.state = dict()
    model.to(args.device)

    if not args.use_ecdf:
        model, best_dict_marg_flow_1, best_dict_marg_flow_2, marg_flow_1_output, marg_flow_2_output = train_marginals(model, disable_tqdm, error_bars, rvine)

        model.marg_flow_1.eval()
        model.marg_flow_2.eval()
    else:
        norm_distr = scipy.stats.norm()
        ecdf_1 = ECDF(dataset.trn[:, 0])
        uniform_1 = ecdf_1(dataset.trn[:, 0])
        uniform_1[uniform_1 == 0] = eps
        uniform_1[uniform_1 == 1] = 1 - eps
        marg_flow_1_output = torch.from_numpy(norm_distr.ppf(uniform_1)).float().reshape(-1, 1)
        ecdf_2 = ECDF(dataset.trn[:, 1])
        uniform_2 = ecdf_2(dataset.trn[:, 1])
        uniform_2[uniform_2 == 0] = eps
        uniform_2[uniform_2 == 1] = 1 - eps
        marg_flow_2_output = torch.from_numpy(norm_distr.ppf(uniform_2)).float().reshape(-1, 1)
    for param in model.cop_flow.parameters():
        param.requires_grad = True

    model, best_dict, test_dict, best_dict_cop_flow = train_copula_flow(model, dataset.trn, disable_tqdm, error_bars, rvine, marg_flow_1_output, marg_flow_2_output)

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   test_dict.items()}  # save test set metrics in dict format
    sep = '_'
    if not args.use_ecdf:
        epochs = sep.join(list([str(best_dict_marg_flow_1['best_validation_epoch']),
                                str(best_dict_marg_flow_2['best_validation_epoch']),
                                str(best_dict_cop_flow['best_validation_epoch'])]))
    else:
        epochs = sep.join(list([str(best_dict_cop_flow['best_validation_epoch'])]))

    if not rvine:
        save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=error_bars, test_epoch=epochs)

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
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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

    # Specify, that this cop_flow is part of a CM_Flow
    args.cop_flow_part_of_CM_Flow = True

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
        train_and_plot(args=args,
                       dataset=dataset,
                       data_loaders=data_loaders,
                       disable_tqdm=False)
