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

from RealNVP_modules.eval import jsd_eval, jsd_graph, margin_uniformity, plot_margins
from RealNVP import build_model as build_model_RealNVP
import RealNVP_modules.flows as fnn
import RealNVP_modules.utils

from DDSF_modules.utils import load_data as load_data_DDSF
from DDSF import build_model as build_model_DDSF
import DDSF_modules.visualizer as visualizer

from utils.save_statistics import save_statistics, save_model, load_model
from utils.loss_plots import collect_experiment_dicts, plot_result_graphs


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


def train(epoch, train_loader, current_epoch_losses, device):
    """Performs training.

    Params:
        epoch: current epoch
        train_loader: data loader
        current_epoch_loss: dictionary containing the training loss of the epoch
        device: device

    Returns:
        current_epoch_losses: updated training loss dictionary
    """
    model.train()

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        losses = model.loss(data)
        loss = 0
        for loss_element in losses:
            loss += loss_element.mean()
        current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list

        loss.backward()
        if args.clip_grad_norm:
            model.clip_grad_norm()

        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(loss))

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    # with torch.no_grad():
    #     model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

    return current_epoch_losses


def validate(epoch, model, loader, device,
             current_epoch_losses=None, best_dict=None):
    """Return log probabilities on validation set.

    Params:
        epoch: epoch to validate
        model: model to validate
        loader: whether to use train/val/test set loader
        device: used device
        current_epoch_losses: dictionary with the current epoch losses
        best_dict: dictionary containing the best validation loss, best validation epoch
                   and best model

    Returns:
        current_epoch_losses: updated current_epoch_losses
        best_dict: updated best_dict
    """
    model.eval()

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            losses = model.loss(data)
            current_loss = 0
            for loss_element in losses:
                current_loss += loss_element.mean()
        if current_epoch_losses is not None:
            current_epoch_losses["val_loss"].append(current_loss)  # add current iter loss to val loss list.
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss < best_dict['best_validation_loss']:  # if current epoch's mean val acc is greater than the saved best val acc then
                best_dict['best_validation_loss'] = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                best_dict['best_validation_epoch'] = epoch  # set the experiment-wise best val idx to be the current epoch's idx
                # best_dict['best_model'] = copy.deepcopy(model)

        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(current_loss))

    pbar.close()
    return current_epoch_losses, best_dict


def test(epoch, model, loader, device,
         current_epoch_test):
    """Return log probabilities on test set.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        current_epoch_test: dictionary with the current epoch test stats

    Returns:
        current_epoch_test: updated current_epoch_test
    """
    model.eval()

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            losses = model.loss(data)
            current_loss = 0
            for loss_element in losses:
                current_loss += loss_element.mean()
        current_epoch_test["test_loss"].append(current_loss)  # add current iter loss to test loss list.

        pbar.update(data.size(0))
        pbar.set_description('Test, Log likelihood in nats in epoch {}: {:.6f}'.format(epoch, np.mean(current_epoch_test["test_loss"])))

    pbar.close()

    return current_epoch_test


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
    dataset, data_loaders = utils.load_data(args)

    # Build model and send to device
    model, model_RealNVP, model_DDSF_1, model_DDSF_2 = build_model(args)
    model.state = dict()
    model_RealNVP.state = dict()

    model.to(args.device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)

    # Save losses and best epoch stats and model in dictionary
    total_losses = {"train_loss": [], "val_loss": []}  # initialize a dict to keep the per-epoch metrics
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0, 'best_model': model}
    current_epoch_test = {"test_loss": [], 'jsd_test': [], 't_1': [], 't_2': [], 'm_1': [], 'm_2': []}  # initialize a statistics dict

    # Train
    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        # Initialize dictionary for epoch losses
        current_epoch_losses = {"train_loss": [], "val_loss": []}

        # Train model
        current_epoch_losses = train(epoch,
                                     data_loaders['train_loader'],
                                     current_epoch_losses,
                                     args.device)

        # Perform Validation
        current_epoch_losses, best_dict = validate(epoch,
                                                   model,
                                                   data_loaders['valid_loader'],
                                                   args.device,
                                                   current_epoch_losses=current_epoch_losses,
                                                   best_dict=best_dict)

        # Set model state to epoch
        model.state['model_epoch'] = epoch

        # save model and best val idx and best val acc, using the model dir, model name and model idx
        save_model(model=model, model_RealNVP=model_RealNVP,
                   model_save_dir=args.experiment_saved_models,
                   model_save_name="train_model", model_idx=epoch,
                   best_validation_model_idx=best_dict['best_validation_epoch'],
                   best_validation_model_loss=best_dict['best_validation_loss'])

        save_model(model=model, model_RealNVP=model_RealNVP,
                   model_save_dir=args.experiment_saved_models,
                   model_save_name="train_model", model_idx=epoch,
                   best_validation_model_idx=best_dict['best_validation_epoch'],
                   best_validation_model_loss=best_dict['best_validation_loss'])

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

        print('Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
              format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

        # Save sample plots every n epochs
        if epoch % args.plot_frequ == 0:
            RealNVP_modules.utils.save_samples_plot(args, epoch, model_RealNVP, dataset)

    # Calculate test statistics
    # Load best validation model
    load_model(model=model, model_RealNVP=model_RealNVP, model_save_dir=args.experiment_saved_models, model_idx=best_dict['best_validation_epoch'],
               model_save_name="train_model")

    # Perform test evaluation
    current_epoch_test = test(best_dict['best_validation_epoch'],
                              model,
                              data_loaders['test_loader'],
                              args.device,
                              current_epoch_test=current_epoch_test)

    # Calculate Jensen-Shannon Divergence of copula
    current_epoch_test = jsd_eval(args,
                                  best_dict['best_validation_epoch'],
                                  model_RealNVP,
                                  data_loaders['test_loader'],
                                  args.device,
                                  current_epoch_test=current_epoch_test)

    # Evaluate copula margins on test set
    current_epoch_test = margin_uniformity(best_dict['best_validation_epoch'],
                                           model_RealNVP,
                                           data_loaders['test_loader'],
                                           args.device,
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
    plot_result_graphs(args.figures_path, args.exp_name, args.copula, result_dict)

    # Plot samples for best epoch
    RealNVP_modules.utils.save_samples_plot(args, best_dict['best_validation_epoch'], model_RealNVP, dataset)

    # Plot Copula Margins
    plot_margins(args,
                 best_dict['best_validation_epoch'],
                 model_RealNVP,
                 data_loaders['test_loader'])

    # Plot Margins
    args.mu = 0
    args.var = 1
    marginal_dataset, __, __ = load_data_DDSF(args)

    #visualizer.visualize1D_CM(dataset, best_dict['best_model'], best_dict['best_validation_epoch'], args, obs=1000)
    #visualizer.visualize1D_CM(dataset, best_dict['best_model'], best_dict['best_validation_epoch'], args, obs=1000)

    # Plot pointwise copula difference
    jsd_graph(args,
              best_dict['best_validation_epoch'],
              model_RealNVP)

