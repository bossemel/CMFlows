import copy

import torch
# import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

# import datasets
import RealNVP_modules.flows as fnn
import RealNVP_modules.utils as utils
from RealNVP_modules.options import TrainOptions
import torch.nn as nn
from RealNVP_modules.eval import jsd_eval, jsd_graph, margin_uniformity, validate
from RealNVP_modules.save_statistics import save_statistics
import os
import numpy as np
from pathlib import Path


def build_model(args, num_cond_inputs, num_inputs, device):
    if args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST']:
        num_hidden = {
            'POWER': 100,
            'GAS': 100,
            'HEPMASS': 512,
            'MINIBOONE': 512,
            'BSDS300': 512,
            'MOONS': 64,
            'GAUSSIAN': 64,
            'CLAYTON': 64,
            'GUMBEL': 64,
            'FRANK': 64,
            'TDISTR': 64,
            'MNIST': 1024
        }[args.dataset]
    else:
        num_hidden = {args.dataset: args.num_hidden}[args.dataset]

    # act = 'tanh' if args.dataset == 'GAS' else 'relu'

    modules = []

    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
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


def train(epoch, train_loader, val_loader, current_epoch_losses):
    global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list

        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device), train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

    return current_epoch_losses


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

    Path(args.figures_path).mkdir(parents=True, exist_ok=True)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    dataset, num_cond_inputs, num_inputs, train_loader, valid_loader, test_loader = utils.load_data(args)

    # Generate the directory names
    experiment_folder = os.path.abspath(args.exp_name)
    experiment_logs = os.path.abspath(os.path.join(experiment_folder, "result_outputs"))
    experiment_saved_models = os.path.abspath(os.path.join(experiment_folder, "saved_models"))

    # Set best models to be at 0 since we are just starting
    best_val_model_idx = 0
    best_val_model_acc = 0.

    if not os.path.exists(experiment_folder):  # If experiment directory does not exist
        os.mkdir(experiment_folder)  # create the experiment directory
        os.mkdir(experiment_logs)  # create the experiment log directory
        os.mkdir(experiment_saved_models)  # create the experiment saved models directory

    total_losses = {"train_loss": [], "val_loss": []}  # initialize a dict to keep the per-epoch metrics

    model = build_model(args, num_cond_inputs, num_inputs, device)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    writer = SummaryWriter(comment='RealNVP' + "_" + args.dataset)
    global_step = 0

    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        current_epoch_losses = {"train_loss": [], "val_loss": []}
        current_epoch_losses = train(epoch, train_loader, valid_loader, current_epoch_losses)
        validation_loss, current_epoch_losses = validate(epoch, model, valid_loader, device, writer, global_step, current_epoch_losses=current_epoch_losses)

        for key, value in current_epoch_losses.items():
            total_losses[key].append(np.mean(
                value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

        save_statistics(experiment_log_dir=experiment_logs, filename='summary.csv',
                        stats_dict=total_losses, current_epoch=epoch,
                        continue_from_mode=epoch)  # save statistics to stats file.

        if args.early_stopping is True:
            if epoch - best_validation_epoch >= 30:
                break

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_validation_epoch, -best_validation_loss))

        if (args.dataset in ['MOONS', 'GAUSSIAN', 'TDISTR', 'CLAYTON', 'FRANK', 'GUMBEL']) and epoch % 10 == 0:
            utils.save_moons_plot(epoch, model, dataset)
        elif args.dataset == 'MNIST' and epoch % 1 == 0:
            utils.save_images(epoch, model, args.cond)

    validate(best_validation_epoch, best_model, test_loader, device, writer, global_step, prefix='Test')

    jsd_eval(args, best_validation_epoch, best_model, test_loader, device, writer, global_step, prefix='Test')

    margin_uniformity(best_validation_epoch, best_model, test_loader, device, writer, global_step, sigmoid=args.sigmoid, prefix='Test')

    jsd_graph(args, best_validation_epoch, best_model, test_loader, writer, global_step, prefix='Test')
