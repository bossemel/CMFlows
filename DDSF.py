import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import random
import copy

import DDSF_modules.visualizer as visualizer
from DDSF_modules import nn_modules as nn_, flows, utils, optim
from DDSF_modules.utils import load_data
from DDSF_modules.options import TrainOptions

from utils.save_statistics import save_statistics
from utils.loss_plots import collect_experiment_dicts


def build_model(args):
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
                       fixed_order=True),
        flows.FlipFlow(1)) for i in range(args.num_flow_layers_DDSF)] + \
        [flows.LinearFlow(args.dim, 1), ]

    model = MAF(*sequels)
    return model


def train(epoch, train_loader, current_epoch_losses):
    model.train()

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(args.device)
        optimizer.zero_grad()

        loss = model.loss(data).mean()
        current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list

        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(loss))

    pbar.close()

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
            current_loss = model.loss(data).mean().item()
        if current_epoch_losses is not None:
            current_epoch_losses["val_loss"].append(current_loss)  # add current iter loss to val loss list.
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss < best_dict['best_validation_loss']:  # if current epoch's mean val acc is greater than the saved best val acc then
                best_dict['best_validation_loss'] = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                best_dict['best_validation_epoch'] = epoch  # set the experiment-wise best val idx to be the current epoch's idx
                best_dict['best_model'] = copy.deepcopy(model)

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
            current_loss = model.loss(data).mean().item()
        current_epoch_test["test_loss"].append(current_loss)  # add current iter loss to test loss list.

        pbar.update(data.size(0))
        pbar.set_description('Test, Log likelihood in nats in epoch {}: {:.6f}'.format(epoch, np.mean(current_epoch_test["test_loss"])))

    pbar.close()

    return current_epoch_test


class MAF(nn.Sequential):

    def get_model(self):
        return self

    def density(self, samples, logdets=None, context=None, zeros=None):
        self.n = args.batch_size
        self.context = Variable(torch.FloatTensor(self.n, 1).zero_()) + 2.0
        self.logdets = Variable(torch.FloatTensor(self.n).zero_())
        self.zeros = Variable(torch.FloatTensor(self.n, 2).zero_())

        logdets = self.logdets if logdets is None else logdets
        context = self.context if context is None else context
        zeros = self.zeros if zeros is None else zeros
        z, logdet, _ = self((samples, logdets, context))
        density = utils.log_normal(z, zeros, zeros + 1.0).sum(1) - logdet
        return density

    def loss(self, x):
        return - self.density(x)

    def sample(self, num_samples=None, noise=None):
        context = Variable(torch.FloatTensor(num_samples, 1).zero_())
        logdet = Variable(torch.FloatTensor(num_samples).zero_())

        if noise is None:
            noise = torch.Tensor(num_samples).normal_().reshape(-1, 1)
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples, __, __ = self((noise, logdet, context))
        return samples

    # def clip_grad_norm(self):
    #     nn.utils.clip_grad_norm_(self.flow.parameters(), self.clip)


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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    # optimizer in train model:
    # optimizer = optim.Adam(model.parameters(),
    # lr=args.lr,
    # betas=(args.beta1, args.beta2),
    # amsgrad=bool(args.amsgrad),
    # polyak=args.polyak)

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
                                                   args.device,
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

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

        # Save sample plots every 10 epochs
        if epoch % args.plot_frequ == 0:
            visualizer.visualize1D(dataset, model, epoch, args)

    # Calculate test statistics
    current_epoch_test = test(best_dict['best_validation_epoch'],
                              best_dict['best_model'],
                              data_loaders['test_loader'],
                              args.device,
                              current_epoch_test=current_epoch_test)

    visualizer.visualize1D(dataset, best_dict['best_model'], best_dict['best_validation_epoch'], args)

    # # Calculate Jensen-Shannon Divergence on test set
    # current_epoch_test = jsd_eval(args,
    #                               best_dict['best_validation_epoch'],
    #                               best_dict['best_model'],
    #                               data_loaders['test_loader'],
    #                               device,
    #                               current_epoch_test=current_epoch_test)

    # # Evaluate margins on test set
    # current_epoch_test = margin_uniformity(best_dict['best_validation_epoch'],
    #                                        best_dict['best_model'],
    #                                        data_loaders['test_loader'],
    #                                        device,
    #                                        transform_fct=args.transform_fct,
    #                                        current_epoch_test=current_epoch_test)

    # Gather test losses and save statistics
    test_losses = {key: [np.mean(value)] for key, value in
                   current_epoch_test.items()}  # save test set metrics in dict format
    save_statistics(experiment_log_dir=args.experiment_logs, filename='test_summary.csv',
                    # save test set metrics on disk in .csv format
                    stats_dict=test_losses, current_epoch=0, continue_from_mode=False, test_epoch=best_dict['best_validation_epoch'])

    # Plot losses
    result_dict = collect_experiment_dicts(target_dir=args.experiment_logs)
    # plot_result_graphs(args.figures_path, args.exp_name, args.dataset, result_dict)

    # Plot samples for best epoch
    # utils.save_samples_plot(args, best_dict['best_validation_epoch'], best_dict['best_model'], dataset)

    # # Plot Margins
    # plot_margins(args,
    #              best_dict['best_validation_epoch'],
    #              best_dict['best_model'],
    #              data_loaders['test_loader'])

    # # Plot pointwise difference
    # jsd_graph(args,
    #           best_dict['best_validation_epoch'],
    #           best_dict['best_model'])
