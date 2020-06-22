import torch
import torch.optim as optim
import torch.utils.data

from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import random

from CM_modules.options import TrainOptions
import CM_modules.utils as utils
import CM_modules.flows as flows
from CM_modules.visualizer import visualize1D_CM, save_samples_plot_copula

from RealNVP_modules.eval import jsd_eval as jsd_eval_copula, jsd_graph, margin_uniformity, plot_margins
from RealNVP import build_model as build_model_RealNVP
import RealNVP_modules.flows as fnn
import RealNVP_modules.utils as RealNVP_utils

from DDSF_modules.utils import load_data as load_data_DDSF, jsd_eval as jsd_eval_marginal
from DDSF import build_model as build_model_DDSF

from utils.save_statistics import save_statistics, save_model, load_model, model_loader
from utils.loss_plots import collect_experiment_dicts, plot_result_graphs
from utils.various import logit
from utils.visualizer import visualize_joint

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

    # Save losses and best epoch stats and model in dictionary
    total_losses = {'train_loss': [],
                    'val_loss': []}  # initialize a dict to keep the per-epoch metrics
    best_dict = {'best_validation_loss': float('inf'),
                 'best_validation_epoch': 0}
    current_epoch_test = {'test_loss': [],
                          'jsd_test_copula': [],
                          'jsd_test_marginal': [],
                          't_1': [],
                          't_2': [],
                          'm_1': [],
                          'm_2': []}  # initialize a statistics dict

    if args.pretrain_models:
        # Train
        args.optimizer = optim.Adam(model_DDSF_1.parameters(), lr=args.lr, betas=args.betas)
        model_DDSF_1, best_dict_DDSF_1, current_epoch_test_DDSF_1 = train_val(current_model=model_DDSF_1,
                                                                              model_name='DDSF_1',
                                                                              args=args,
                                                                              data_loaders=data_loaders,
                                                                              dataset=dataset)
        args.optimizer = optim.Adam(model_DDSF_2.parameters(), lr=args.lr, betas=args.betas)
        model_DDSF_2, best_dict_DDSF_2, current_epoch_test_DDSF_2 = train_val(current_model=model_DDSF_2,
                                                                              model_name='DDSF_2',
                                                                              args=args,
                                                                              data_loaders=data_loaders,
                                                                              dataset=dataset)

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

        model_RealNVP, best_dict_RealNVP, current_epoch_test_RealNVP = train_val(model_RealNVP,
                                                                                 model_name='RealNVP',
                                                                                 args=args,
                                                                                 data_loaders=data_loaders,
                                                                                 dataset=dataset,
                                                                                 transform_model_1=model_DDSF_1,
                                                                                 transform_model_2=model_DDSF_1)
        best_dict = best_dict_RealNVP
        current_epoch_test = current_epoch_test_RealNVP

        output_copula = model_RealNVP.sample_copula(num_samples=1000)
        visualize_joint(output_copula.detach().numpy(), args, name='output_copula')

    # # Train
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    model, best_dict, current_epoch_test = train_val(model,
                                                     model_name='CM_Flow',
                                                     args=args,
                                                     data_loaders=data_loaders,
                                                     dataset=dataset)

    output_copula = model.sample_copula(num_samples=1000)
    visualize_joint(output_copula.detach().numpy(), args, name='output_copula')

    output_copula = model.sample(num_samples=1000)
    visualize_joint(output_copula.detach().numpy(), args, name='output_copula_normal')

    # for epoch in range(args.epochs):
    #     print('\nEpoch: {}'.format(epoch))

    #     # Initialize dictionary for epoch losses
    #     current_epoch_losses = {"train_loss": [], "val_loss": []}

    #     # Train model
    #     current_epoch_losses = train(epoch,
    #                                  data_loaders['train_loader'],
    #                                  current_epoch_losses,
    #                                  args.device)

    #     # Perform Validation
    #     current_epoch_losses, best_dict = validate(epoch,
    #                                                data_loaders['valid_loader'],
    #                                                args.device,
    #                                                current_epoch_losses=current_epoch_losses,
    #                                                best_dict=best_dict)

    #     # Set model state to epoch
    #     model.state['model_epoch'] = epoch
    #     model_RealNVP.state['model_epoch'] = epoch
    #     model_DDSF_1.state['model_epoch'] = epoch
    #     model_DDSF_2.state['model_epoch'] = epoch

    #     # save model and best val idx and best val acc, using the model dir, model name and model idx
    #     save_model(model=model, model_RealNVP=model_RealNVP, model_DDSF_1=model_DDSF_1, model_DDSF_2=model_DDSF_2,
    #                model_save_dir=args.experiment_saved_models,
    #                model_save_name="train_model", model_idx=epoch,
    #                best_validation_model_idx=best_dict['best_validation_epoch'],
    #                best_validation_model_loss=best_dict['best_validation_loss'])

    #     save_model(model=model, model_RealNVP=model_RealNVP, model_DDSF_1=model_DDSF_1, model_DDSF_2=model_DDSF_2,
    #                model_save_dir=args.experiment_saved_models,
    #                model_save_name="train_model", model_idx=epoch,
    #                best_validation_model_idx=best_dict['best_validation_epoch'],
    #                best_validation_model_loss=best_dict['best_validation_loss'])

    #     # Save mean of each epoch in total losses dictionary
    #     for key, value in current_epoch_losses.items():
    #         total_losses[key].append(np.mean(
    #             value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

    #     # Save current epoch statistics
    #     save_statistics(experiment_log_dir=args.experiment_logs, filename='summary.csv',
    #                     stats_dict=total_losses, current_epoch=epoch,
    #                     continue_from_mode=epoch)  # save statistics to stats file.

    #     # Early stopping
    #     if args.early_stopping is True:
    #         if epoch - best_dict['best_validation_epoch'] >= 30:
    #             break

    #     print('Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
    #           format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

    #     # Save sample plots every n epochs
    #     if epoch % args.plot_frequ == 0:
    #         save_samples_plot_copula(args=args,
    #                                  epoch=epoch,
    #                                  model=model_RealNVP,
    #                                  dataset=dataset)

    # # Calculate test statistics
    # # Load best validation model
    # load_model(model=model, model_RealNVP=model_RealNVP, model_DDSF_1=model_DDSF_1, model_DDSF_2=model_DDSF_2,
    #            model_save_dir=args.experiment_saved_models, model_idx=best_dict['best_validation_epoch'],
    #            model_save_name="train_model")

    # # Perform test evaluation
    # current_epoch_test = test(best_dict['best_validation_epoch'],
    #                           model,
    #                           data_loaders['test_loader'],
    #                           args.device,
    #                           current_epoch_test=current_epoch_test)

    # # Calculate Jensen-Shannon Divergence of copula
    # current_epoch_test = jsd_eval_copula(args,
    #                                      best_dict['best_validation_epoch'],
    #                                      model_RealNVP,
    #                                      data_loaders['test_loader'],
    #                                      args.device,
    #                                      current_epoch_test=current_epoch_test,
    #                                      cm_flow=True)

    # # Calculate Jensen-Shannon Divergence of marginal 1
    # args.marginal = args.marginal_1
    # current_epoch_test = jsd_eval_marginal(marginal=args.marginal_1,
    #                                        args=args,
    #                                        epoch=best_dict['best_validation_epoch'],
    #                                        model=model_DDSF_1,
    #                                        current_epoch_test=current_epoch_test)

    # # Calculate Jensen-Shannon Divergence of marginal 1
    # args.marginal = args.marginal_2
    # current_epoch_test = jsd_eval_marginal(marginal=args.marginal_2,
    #                                        args=args,
    #                                        epoch=best_dict['best_validation_epoch'],
    #                                        model=model_DDSF_2,
    #                                        current_epoch_test=current_epoch_test)

    # # Evaluate copula margins on test set
    # current_epoch_test = margin_uniformity(best_dict['best_validation_epoch'],
    #                                        model_RealNVP,
    #                                        data_loaders['test_loader'],
    #                                        args.device,
    #                                        transform_fct=args.transform_fct,
    #                                        current_epoch_test=current_epoch_test,
    #                                        cm_flow=True)

    # # Gather test losses and save statistics
    # test_losses = {key: [np.mean(value)] for key, value in
    #                current_epoch_test.items()}  # save test set metrics in dict format
    # save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
    #                 # save test set metrics on disk in .csv format
    #                 stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # # Plot losses
    # result_dict = collect_experiment_dicts(target_dir=args.experiment_logs, model_name='')
    # # plot_result_graphs(args.figures_path, args.exp_name, args.copula, result_dict)

    # # Plot samples for best epoch
    # save_samples_plot_copula(args, best_dict['best_validation_epoch'], model_RealNVP, dataset)

    # # Plot Copula Margins
    # plot_margins(args,
    #              best_dict['best_validation_epoch'],
    #              model_RealNVP,
    #              data_loaders['test_loader'])

    # # Plot Margins
    # args.mu = 0
    # args.var = 1

    # args.marginal = args.marginal_1
    # marginal_dataset_1, __, __ = load_data_DDSF(args)
    # args.marginal = args.marginal_2
    # marginal_dataset_2, __, __ = load_data_DDSF(args)

    # visualize1D_CM(marginal_dataset_1, model_DDSF_1, best_dict['best_validation_epoch'], args, obs=1000)
    # visualize1D_CM(marginal_dataset_2, model_DDSF_2, best_dict['best_validation_epoch'], args, obs=1000)

    # # Plot pointwise copula difference
    # jsd_graph(args,
    #           best_dict['best_validation_epoch'],
    #           model_RealNVP)

