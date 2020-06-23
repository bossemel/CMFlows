import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from pathlib import Path
import random

from DDSF_modules import nn_modules as nn_, flows, optim
from DDSF_modules.utils import load_data
from DDSF_modules.options import TrainOptions
from DDSF_modules.flows import MAF

from experiment_runner import train_val
from utils.save_statistics import save_statistics


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
    dataset, num_inputs, data_loaders = load_data(args)

    # Build model and send to device
    model = build_model(args)
    model.state = dict()
    model.to(args.device)

    # optimizer in MAF:
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=1e-6)
    # optimizer in train model:
    # optimizer = optim.Adam(model.parameters(),
    # lr=args.lr,
    # betas=(args.beta1, args.beta2),
    # amsgrad=bool(args.amsgrad),
    # polyak=args.polyak)

    # Save losses and best epoch stats and model in dictionary
    total_losses = {"train_loss": [], "val_loss": []}  # initialize a dict to keep the per-epoch metrics
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0}
    test_dict = {"test_loss": [], 'jsd_test_marginal': []}  # initialize a statistics dict

    # Train
    model, best_dict, test_dict = train_val(current_model=model,
                                            model_name='DDSF',
                                            args=args,
                                            data_loaders=data_loaders,
                                            dataset=dataset)

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   test_dict.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # for epoch in range(args.epochs):
    #     print('\nEpoch: {}'.format(epoch))

    #     current_epoch_losses = {"train_loss": [], "val_loss": []}
    #     current_epoch_losses = train(epoch=epoch,
    #                                  model=model,
    #                                  train_loader=data_loaders['train_loader'],
    #                                  current_epoch_losses=current_epoch_losses,
    #                                  device=args.device)
    #     current_epoch_losses, best_dict = validate(epoch,
    #                                                model,
    #                                                data_loaders['valid_loader'],
    #                                                args.device,
    #                                                current_epoch_losses=current_epoch_losses,
    #                                                best_dict=best_dict)

    #     model.state['model_epoch'] = epoch

    #     # save model and best val idx and best val acc, using the model dir, model name and model idx
    #     save_model(model=model,
    #                model_save_dir=args.experiment_saved_models,
    #                model_save_name="train_model", model_idx=epoch,
    #                best_validation_model_idx=best_dict['best_validation_epoch'],
    #                best_validation_model_loss=best_dict['best_validation_loss'])

    #     save_model(model=model,
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

    #     print(
    #         'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
    #         format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

    #     # Save sample plots every 10 epochs
    #     if epoch % args.plot_frequ == 0:
    #         visualizer.visualize1D(marginal=dataset, model=model, epoch=epoch, args=args, obs=10000)

    # Calculate test statistics
    # load best validation model
    # load_model(model=model, model_save_dir=args.experiment_saved_models, model_idx=best_dict['best_validation_epoch'],
    #            model_save_name="train_model")

    # test_dict = test(best_dict['best_validation_epoch'],
    #                           model,
    #                           data_loaders['test_loader'],
    #                           args.device,
    #                           test_dict=test_dict)

    # # Calculate Jensen-Shannon Divergence of copula
    # test_dict = jsd_eval(marginal=dataset,
    #                               args=args,
    #                               epoch=best_dict['best_validation_epoch'],
    #                               model=model,
    #                               test_dict=test_dict)

    # visualizer.visualize1D(dataset, model, best_dict['best_validation_epoch'], args, obs=10000, best_val=True)

    # # Gather test losses and save statistics
    # test_losses = {key: [np.mean(value)] for key, value in
    #                test_dict.items()}  # save test set metrics in dict format
    # save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
    #                 # save test set metrics on disk in .csv format
    #                 stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # # Plot losses
    # result_dict = collect_experiment_dicts(target_dir=args.experiment_logs)
    # plot_result_graphs(args.figures_path, args.exp_name, args.marginal, result_dict)
