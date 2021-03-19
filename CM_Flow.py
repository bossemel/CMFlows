import torch
import torch.optim as optim
import torch.utils.data
import os
import numpy as np
import random
import csv
import json
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats

from CM_modules.options import TrainOptions
import CM_modules.utils as utils
import CM_modules.flows as flows
from utils.visualizer import visualize_joint
from utils.load_and_save import save_statistics, load_statistics, load_model
from experiment_runner import train_val
from utils import create_folders
eps = 1e-09


def build_model(args_):
    """Builds the CM Flow model. It is a concatenation of cop_flow and marg_flow.

    Params:
        args_: passed option arguments

    Returns:
        model: CM Flows model
        cop_flow: cop_flow model
        marg_flow_1: 1st marg_flow model
        marg_flow_2: 2nd marg_flow model
    """
    model_ = flows.CMFlow(args=args_)

    return model_


def visualize_cop_flow_output(model, args_):
    viz_obs = 100000
    with torch.no_grad():
        if args_.conditional_copula:
            context = torch.tensor(np.random.normal(size=(viz_obs, 1))).float()
            output_copula = model.cop_flow.sample_copula(num_samples=viz_obs, context=context,
                                                         device=args_.device).cpu()
            visualize_joint(output_copula, args_.figures_path, name='output_copula_test')
            output_copula = model.cop_flow.sample(num_samples=viz_obs, context=context, device=args_.device).cpu()
            visualize_joint(output_copula, args_.figures_path, name='output_copula_untransformed_test')
        else:
            output_copula = model.cop_flow.sample_copula(num_samples=viz_obs, device=args_.device).cpu()
            visualize_joint(output_copula, args_.figures_path, name='output_copula_test')
            output_copula = model.cop_flow.sample(num_samples=viz_obs, device=args_.device).cpu()
            visualize_joint(output_copula, args_.figures_path, name='output_copula_untransformed_test')


def batch_transform(batch_size, transform_fct, dataset):
    split = torch.split(dataset, dataset.shape[0] % batch_size, dim=0)
    split = torch.cat([transform_fct(sp) for sp in split], dim=0)
    assert split.shape == dataset.shape
    return split


def marg_flow_transform(args_, model, dataset, dim):
    with torch.no_grad():
        if args_.marg_flow == 'DDSF':
            marg_flow_output_trn = batch_transform(batch_size=args_.batch_size,
                                                   transform_fct=model.transform_to_noise,
                                                   dataset=dataset[0][:, dim:dim + 1].to(args_.device).detach().clone())
            marg_flow_output_val = batch_transform(batch_size=args_.batch_size,
                                                   transform_fct=model.transform_to_noise,
                                                   dataset=dataset[1][:, dim:dim + 1].to(args_.device).detach().clone())
            marg_flow_output_tst = batch_transform(batch_size=args_.batch_size,
                                                   transform_fct=model.transform_to_noise,
                                                   dataset=dataset[2][:, dim:dim + 1].to(args_.device).detach().clone())
        else:
            marg_flow_output_trn = model.flow.transform_to_noise(dataset[0][:, dim:dim + 1].to(args_.device)
                                                                 .detach().clone()).reshape(-1, 1)
            marg_flow_output_val = model.flow.transform_to_noise(dataset[1][:, dim:dim + 1].to(args_.device)
                                                                 .detach().clone()).reshape(-1, 1)
            marg_flow_output_tst = model.flow.transform_to_noise(dataset[2][:, dim:dim + 1].to(args_.device)
                                                                 .detach().clone()).reshape(-1, 1)
    visualize_joint(torch.cat([marg_flow_output_trn, marg_flow_output_trn], dim=1).cpu(), args_.figures_path,
                    name='marg_{}_transform_output'.format(dim))
    return marg_flow_output_trn, marg_flow_output_val, marg_flow_output_tst


def marginal_flow_train(model, args_, name, dim, dataset, data_loaders, disable_tqdm, error_bars):
    model_dict = {}
    best_loss = 1000
    for jj in range(3):
        model.init_marg_flow()
        model.marg_flow.state = dict()
        args_.clip_grad_norm = args_.clip_grad_norm_m
        args_.optimizer, args_.scheduler = set_optimizer_scheduler(model.marg_flow,
                                                                   args_.lr_m,
                                                                   args_.weight_decay_m,
                                                                   args_.amsgrad_m,
                                                                   args_.epochs)
        args_.clip = args_.clip_c
        current_name = name + '_' + str(jj)
        best_dict_marg_flow, __ = train_val(model=model.marg_flow,
                                            model_name=current_name,
                                            args=args_,
                                            data_loaders=data_loaders,
                                            disable_tqdm=disable_tqdm,
                                            error_bars=error_bars,
                                            save_name=str(jj))
        model_dict[current_name] = model.marg_flow
        if best_dict_marg_flow['best_validation_loss'] < best_loss:
            best_try = str(jj)
            best_model = current_name
            best_epoch = best_dict_marg_flow['best_validation_epoch']
            best_loss = best_dict_marg_flow['best_validation_loss']

    model.marg_flow = load_model(model=model_dict[best_model], model_save_dir=args_.experiment_saved_models,
                                 model_save_name='best_epoch_model' + best_try, model_idx=best_epoch)
    model.marg_flow.eval()
    marg_flow_output = marg_flow_transform(args_=args_, model=model.marg_flow, dataset=dataset, dim=dim)
    return marg_flow_output, best_dict_marg_flow


def set_optimizer_scheduler(model, lr, weight_decay, amsgrad, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler


def train_marginals(model, args_, data_loaders, dataset, disable_tqdm, error_bars):
    # Pretrain models individually, with cop_flow using the outputs of marg_flow as inputs
    # Train marg_flows
    args_.epochs = args_.epochs_m

    for param in model.marg_flow_2.parameters():
        param.requires_grad = False
    for param in model.cop_flow.parameters():
        param.requires_grad = False
    marg_flow_1_output, best_dict_marg_flow_1 = marginal_flow_train(model=model,
                                                                    args_=args_,
                                                                    name='marg_flow_1',
                                                                    dim=0,
                                                                    dataset=dataset, data_loaders=data_loaders,
                                                                    disable_tqdm=disable_tqdm, error_bars=error_bars)
    for param in model.marg_flow_1.parameters():
        param.requires_grad = False
    for param in model.marg_flow_2.parameters():
        param.requires_grad = True

    marg_flow_2_output, best_dict_marg_flow_2 = marginal_flow_train(model=model, args_=args_, name='marg_flow_2', dim=1,
                                                                    dataset=dataset, data_loaders=data_loaders,
                                                                    disable_tqdm=disable_tqdm, error_bars=error_bars)
    for param in model.marg_flow_2.parameters():
        param.requires_grad = False

    return model, best_dict_marg_flow_1, best_dict_marg_flow_2, marg_flow_1_output, marg_flow_2_output


def transform_dataset(marg_flow_1_output, marg_flow_2_output):
    data_loaders = {}
    with torch.no_grad():
        train_dataset = torch.cat((marg_flow_1_output[0], marg_flow_2_output[0]), dim=1).detach().clone().cpu()
        valid_dataset = torch.cat((marg_flow_1_output[1], marg_flow_2_output[1]), dim=1).detach().clone().cpu()
        test_dataset = torch.cat((marg_flow_1_output[2], marg_flow_2_output[2]), dim=1).detach().clone().cpu()
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs)

        data_loaders['train_loader'] = train_loader
        data_loaders['valid_loader'] = valid_loader
        data_loaders['test_loader'] = test_loader

        normal_distr = torch.distributions.normal.Normal(0, 1)
        train_dataset_uniform = normal_distr.cdf(train_dataset.cpu().clone())
        visualize_joint(train_dataset, args.figures_path,
                        name='marg_flow_transform_output' if not args.use_ecdf else 'ecdf_output')
        visualize_joint(train_dataset_uniform, args.figures_path,
                        name='marg_flow_transform_output_uniform' if not args.use_ecdf else 'ecdf_output_uniform')
    return data_loaders


def train_copula_flow(model, args_, disable_tqdm, error_bars,
                      marg_flow_1_output, marg_flow_2_output):
    for param in model.cop_flow.parameters():
        param.requires_grad = True
    # Train cop_flow
    data_loaders = transform_dataset(marg_flow_1_output=marg_flow_1_output,
                                     marg_flow_2_output=marg_flow_2_output)

    args_.epochs = args_.epochs_c
    args_.optimizer, args_.scheduler = set_optimizer_scheduler(model.cop_flow,
                                                               args_.lr_c,
                                                               args_.weight_decay_c,
                                                               args_.amsgrad_c,
                                                               args_.epochs)
    args_.clip = args_.clip_c
    args_.clip_grad_norm = args_.clip_grad_norm_c
    best_dict_cop_flow, test_dict = train_val(model.cop_flow,
                                              model_name='cop_flow',
                                              args=args_,
                                              data_loaders=data_loaders,
                                              disable_tqdm=disable_tqdm,
                                              error_bars=error_bars)
    model.cop_flow = load_model(model=model.cop_flow, model_save_dir=args_.experiment_saved_models,
                                model_save_name='best_epoch_model',
                                model_idx=best_dict_cop_flow['best_validation_epoch'])
    model.cop_flow.eval()

    if not error_bars:
        with torch.no_grad():
            visualize_cop_flow_output(model, args_)
    return model, test_dict, best_dict_cop_flow


def ecdf_transform(dataset):
    norm_distr = scipy.stats.norm()
    ecdf = ECDF(dataset.clone())
    uniform_samples = ecdf(dataset.clone())
    uniform_samples[uniform_samples == 0] = eps
    uniform_samples[uniform_samples == 1] = 1 - eps
    gaussian_samples = norm_distr.ppf(uniform_samples)
    gaussian_samples = torch.from_numpy(gaussian_samples).float().reshape(-1, 1)
    return gaussian_samples


def train_and_plot(args_, disable_tqdm=False, error_bars=False, rvine=False):
    """Trains the CM Flow, saves test set results and plots.

    Params:
        args_: passed arguments
        disable_tqdm: indicate whether to print progress bar
        error_bars: disables plotting and test set results for error bar calculation
        rvine: disables plotting and test set results for r-vine estimation
    """
    # Set up data loader
    dataset, data_loaders = utils.load_data(args_)

    if not error_bars:
        visualize_joint(dataset[0], args_.figures_path, name='input_dataset')

    # Build model and send to device
    model = build_model(args_)
    model.cop_flow.state = dict()
    model.to(args_.device)

    if not args_.use_ecdf:
        marginal_outputs_ = train_marginals(model=model,
                                            args_=args_,
                                            data_loaders=data_loaders,
                                            dataset=dataset,
                                            disable_tqdm=disable_tqdm,
                                            error_bars=error_bars)
        model, best_dict_marg_flow_1, best_dict_marg_flow_2, marg_flow_1_output, marg_flow_2_output = marginal_outputs_
    else:
        marg_flow_1_output_trn = ecdf_transform(dataset[0][:, 0])
        marg_flow_1_output_val = ecdf_transform(dataset[1][:, 0])
        marg_flow_1_output_tst = ecdf_transform(dataset[2][:, 0])
        marg_flow_1_output = marg_flow_1_output_trn, marg_flow_1_output_val, marg_flow_1_output_tst
        marg_flow_2_output_trn = ecdf_transform(dataset[0][:, 1])
        marg_flow_2_output_val = ecdf_transform(dataset[1][:, 1])
        marg_flow_2_output_tst = ecdf_transform(dataset[2][:, 1])
        marg_flow_2_output = marg_flow_2_output_trn, marg_flow_2_output_val, marg_flow_2_output_tst

    for param in model.cop_flow.parameters():
        param.requires_grad = True

    model, test_dict, best_dict_cop_flow = train_copula_flow(model=model,
                                                             args_=args_,
                                                             disable_tqdm=disable_tqdm,
                                                             error_bars=error_bars,
                                                             marg_flow_1_output=marg_flow_1_output,
                                                             marg_flow_2_output=marg_flow_2_output)

    # Gather test losses and save statistics
    test_losses = {kk: [torch.mean(torch.tensor(value)).item()] for kk, value in
                   test_dict.items()}  # save test set metrics in dict format
    sep = '_'
    if not args_.use_ecdf:
        epochs = sep.join(list([str(best_dict_marg_flow_1['best_validation_epoch']),
                                str(best_dict_marg_flow_2['best_validation_epoch']),
                                str(best_dict_cop_flow['best_validation_epoch'])]))
    else:
        epochs = sep.join(list([str(best_dict_cop_flow['best_validation_epoch'])]))

    if not rvine:
        save_statistics(experiment_log_dir=args_.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=error_bars, test_epoch=epochs)


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    create_folders(args)
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

    # Specify, that this cop_flow is part of a CM_Flow
    args.cop_flow_part_of_CM_Flow = True

    # Train model with specified options
    if args.error_bars is True:
        eval_dict = {}
        train_and_plot(args_=args,
                       disable_tqdm=False,
                       error_bars=False)
        for ii in range(1, 10):
            train_and_plot(args_=args,
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
        train_and_plot(args_=args,
                       disable_tqdm=False)
