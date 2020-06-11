import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import RealNVP_modules.flows as fnn
import RealNVP_modules.utils as utils
from RealNVP_modules.options import TrainOptions
import torch.nn as nn
from RealNVP_modules.eval import jsd_eval, jsd_graph, margin_uniformity, validate, test, plot_margins
from RealNVP_modules.save_statistics import save_statistics
import os
import numpy as np
from pathlib import Path
from RealNVP_modules.loss_plots import collect_experiment_dicts, plot_result_graphs
import random


def build_model(args, num_inputs, device):
    num_hidden = {args.dataset: args.num_hidden}[args.dataset]

    modules = []

    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    return model


def train(epoch, train_loader, current_epoch_losses):
    model.train()

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data).mean()
        current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list

        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(loss))

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

    return current_epoch_losses


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)

    # # Whether to use sigmoidal function before and after RealNVP
    # args.sigmoid = not args.no_sigmoid

    # Cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set up data loader
    dataset, num_inputs, data_loaders = utils.load_data(args)

    # Build model and send to device
    model = build_model(args, num_inputs, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Save losses and best epoch stats and model in dictionary
    total_losses = {"train_loss": [], "val_loss": []}  # initialize a dict to keep the per-epoch metrics
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0, 'best_model': model}
    current_epoch_test = {"test_loss": [], 'jsd_test': [], 't_1': [], 't_2': [], 'm_1': [], 'm_2': []}  # initialize a statistics dict

    # Train
    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        current_epoch_losses = {"train_loss": [], "val_loss": []}
        current_epoch_losses = train(epoch, data_loaders['train_loader'], current_epoch_losses)
        current_epoch_losses, best_dict = validate(epoch,
                                                   model,
                                                   data_loaders['valid_loader'],
                                                   device,
                                                   current_epoch_losses=current_epoch_losses,
                                                   best_dict=best_dict)

        # Save mean of each epoch in total losses dictionary
        for key, value in current_epoch_losses.items():
            total_losses[key].append(np.mean(
                value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

        # Save current epoch statistics
        save_statistics(experiment_log_dir=args.experiment_logs, filename='summary.csv',
                        stats_dict=total_losses, current_epoch=epoch,
                        continue_from_mode=epoch)  # save statistics to stats file.

        # Early stopping
        if args.early_stopping is True:
            if epoch - best_dict['best_validation_epoch'] >= 30:
                break

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

        # Save sample plots every 10 epochs
        if epoch % args.plot_frequ == 0:
            utils.save_samples_plot(args, epoch, model, dataset)

    # Calculate test statistics
    current_epoch_test = test(best_dict['best_validation_epoch'],
                              best_dict['best_model'],
                              data_loaders['test_loader'],
                              device,
                              current_epoch_test=current_epoch_test)

    # Calculate Jensen-Shannon Divergence on test set
    current_epoch_test = jsd_eval(args,
                                  best_dict['best_validation_epoch'],
                                  best_dict['best_model'],
                                  data_loaders['test_loader'],
                                  device,
                                  current_epoch_test=current_epoch_test)

    # Evaluate margins on test set
    current_epoch_test = margin_uniformity(best_dict['best_validation_epoch'],
                                           best_dict['best_model'],
                                           data_loaders['test_loader'],
                                           device,
                                           transform_fct=args.transform_fct,
                                           current_epoch_test=current_epoch_test)

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   current_epoch_test.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # Plot losses
    result_dict = collect_experiment_dicts(target_dir=args.experiment_logs)
    plot_result_graphs(args.figures_path, args.exp_name, args.dataset, result_dict)

    # Plot samples for best epoch
    utils.save_samples_plot(args, best_dict['best_validation_epoch'], best_dict['best_model'], dataset)

    # Plot Margins
    plot_margins(args,
                 best_dict['best_validation_epoch'],
                 best_dict['best_model'],
                 data_loaders['test_loader'])

    # Plot pointwise difference
    jsd_graph(args,
              best_dict['best_validation_epoch'],
              best_dict['best_model'])
