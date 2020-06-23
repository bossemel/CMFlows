import torch
import torch.utils.data

from tqdm import tqdm
import numpy as np

import RealNVP_modules.flows as fnn

from utils.save_statistics import save_statistics, save_model, model_loader
from utils.loss_plots import collect_experiment_dicts, plot_result_graphs
from RealNVP_modules.eval import jsd_eval as jsd_eval_copula, margin_uniformity, plot_margins
from DDSF_modules.utils import jsd_eval as jsd_eval_marginal

from CM_modules.flows import flow_density

eps = 0.0001


def empty_logdets_context(inputs, device):
    n = inputs.shape[0]
    context = torch.FloatTensor(n, 1).zero_().to(device)
    logdets = torch.FloatTensor(n).zero_().to(device)
    return logdets, context


def train(args, epoch, model, train_loader, current_epoch_losses, device,
          model_name=None, transform_model_1=None, transform_model_2=None, transform_inputs=True, disable=False):
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

    pbar = tqdm(total=len(train_loader.dataset), disable=disable)
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
            if model_name == 'DDSF_1':
                data = data[:, 0].reshape(-1, 1)
            elif model_name == 'DDSF_2':
                data = data[:, 1].reshape(-1, 1)
            elif model_name == 'RealNVP' and transform_inputs is True:
                logdets, context = empty_logdets_context(data, device)
                data_1, __, __ = transform_model_1((data[:, 0].reshape(-1, 1), logdets, context))
                data_2, __, __ = transform_model_2((data[:, 1].reshape(-1, 1), logdets, context))
                data = torch.cat((data_1, data_2), dim=1)

        data = data.to(device)
        args.optimizer.zero_grad()

        if model_name == 'CM_Flow':
            logdets, context = empty_logdets_context(data, device)

            output_DDSF_1, logdets_DDSF_1 = model.forward_DDSF_1((data, logdets, context))
            output_DDSF_2, logdets_DDSF_2 = model.forward_DDSF_2((data, logdets, context))
            outputs_DDSFs = torch.cat((output_DDSF_1, output_DDSF_2), dim=1)
            logdets_DDSFs = torch.cat((logdets_DDSF_1.reshape(-1, 1), logdets_DDSF_2.reshape(-1, 1)), dim=1)

            output_RealNVP, logdets_RealNVP = model.forward_RealNVP(outputs_DDSFs, logdets_DDSFs, mode='direct')

            loss_DDSF_1 = -flow_density(output_DDSF_1, logdets_DDSF_1).mean()
            loss_DDSF_2 = -flow_density(output_DDSF_2, logdets_DDSF_2).mean()
            loss_RealNVP = -flow_density(output_RealNVP, logdets_RealNVP).mean()

            loss_DDSF_1.backward(retain_graph=True)
            loss_DDSF_2.backward(retain_graph=True)
            loss_RealNVP.backward()
            loss = loss_RealNVP

            if 'train_loss' in current_epoch_losses:
                current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list
            else:
                current_epoch_losses['train_loss'] = []
            # for loss_element in losses:
            #     loss += loss_element.mean()
        else:
            losses = model.loss(data)
            loss = 0
            loss = losses.mean()
            if 'train_loss' in current_epoch_losses:
                current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list
            else:
                current_epoch_losses['train_loss'] = [loss.item()]

            loss.backward()
        # if model_name in ['DDSF_1', 'DDSF_2', 'CM_Flow']:
        #     if args.clip_grad_norm:
        #         model.clip_grad_norm()
        args.optimizer.step()
        args.optimizer.zero_grad()

        pbar.update(data.size(0))
        pbar.set_description('{} Train, Log likelihood: {:.6f}'.format(model_name, loss))

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if model_name == 'RealNVP':
        if transform_inputs is True:
            with torch.no_grad():
                logdets, context = empty_logdets_context(train_loader.dataset.tensors[0], device)

                data_1, __, __ = transform_model_1((train_loader.dataset.tensors[0][:, 0].reshape(-1, 1), logdets, context))
                data_2, __, __ = transform_model_2((train_loader.dataset.tensors[0][:, 1].reshape(-1, 1), logdets, context))
                train_loader_tensor = torch.cat((data_1, data_2), dim=1)
                model(train_loader_tensor.to(data.device))
        else:
            model(train_loader.dataset.tensors[0].to(data.device))
    elif model_name == 'CM_Flow':
        with torch.no_grad():
            logdets, context = empty_logdets_context(train_loader.dataset.tensors[0], device)

            model((train_loader.dataset.tensors[0], logdets, context))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

    return current_epoch_losses


def validate(epoch, model, loader, device,
             current_epoch_losses=None, best_dict=None, model_name=None,
             transform_model_1=None, transform_model_2=None, transform_inputs=True, disable=False):
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

    pbar = tqdm(total=len(loader.dataset), disable=disable)
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
            if model_name == 'DDSF_1':
                data = data[:, 0].reshape(-1, 1)
            elif model_name == 'DDSF_2':
                data = data[:, 1].reshape(-1, 1)
            elif model_name == 'RealNVP' and transform_inputs is True:
                logdets, context = empty_logdets_context(data, device)
                data_1, __, __ = transform_model_1((data[:, 0].reshape(-1, 1), logdets, context))
                data_2, __, __ = transform_model_2((data[:, 1].reshape(-1, 1), logdets, context))
                data = torch.cat((data_1, data_2), dim=1)
        data = data.to(device)

        with torch.no_grad():
            losses = model.loss(data)
            current_loss = 0
            if model_name == 'CM_Flow':
                logdets, context = empty_logdets_context(data, device)
                output_DDSF_1, logdets_DDSF_1 = model.forward_DDSF_1((data, logdets, context))
                output_DDSF_2, logdets_DDSF_2 = model.forward_DDSF_2((data, logdets, context))
                outputs_DDSFs = torch.cat((output_DDSF_1, output_DDSF_2), dim=1)
                logdets_DDSFs = torch.cat((logdets_DDSF_1.reshape(-1, 1), logdets_DDSF_2.reshape(-1, 1)), dim=1)

                output_RealNVP, logdets_RealNVP = model.forward_RealNVP(outputs_DDSFs, logdets_DDSFs, mode='direct')

                loss_RealNVP = -flow_density(output_RealNVP, logdets_RealNVP).mean()

                current_loss = loss_RealNVP

                # for loss_element in losses:
                #     current_loss += loss_element.mean()
            else:
                current_loss = losses.mean()
        if 'val_loss' in current_epoch_losses:
            current_epoch_losses["val_loss"].append(current_loss.item())  # add current iter loss to val loss list.
        else:
            current_epoch_losses["val_loss"] = [current_loss.item()]  # add current iter loss to val loss list.
        val_mean_loss = np.mean(current_epoch_losses['val_loss'])
        if val_mean_loss < best_dict['best_validation_loss']:  # if current epoch's mean val acc is greater than the saved best val acc then
            best_dict['best_validation_loss'] = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
            best_dict['best_validation_epoch'] = epoch  # set the experiment-wise best val idx to be the current epoch's idx
            # best_dict['best_model'] = copy.deepcopy(model)

        pbar.update(data.size(0))
        pbar.set_description('{} Val, Log likelihood: {:.6f}'.format(model_name, val_mean_loss))

    pbar.close()
    return current_epoch_losses, best_dict


def test(epoch, model, loader, device,
         test_dict, model_name,
         transform_model_1=None, transform_model_2=None, transform_inputs=True,
         disable=False):
    """Return log probabilities on test set.

    Params:
        epoch: best validation epoch
        model: best validation model
        loader: whether to use train/val/test set loader
        device: used device
        test_dict: dictionary with the current epoch test stats

    Returns:
        test_dict: updated test_dict
    """
    model.eval()

    pbar = tqdm(total=len(loader.dataset), disable=disable)
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
            if model_name == 'DDSF_1':
                data = data[:, 0].reshape(-1, 1)
            elif model_name == 'DDSF_2':
                data = data[:, 1].reshape(-1, 1)
            elif model_name == 'RealNVP' and transform_inputs is True:
                logdets, context = empty_logdets_context(data, device)

                data_1, __, __ = transform_model_1((data[:, 0].reshape(-1, 1), logdets, context))
                data_2, __, __ = transform_model_2((data[:, 1].reshape(-1, 1), logdets, context))
                data = torch.cat((data_1, data_2), dim=1)
        data = data.to(device)
        with torch.no_grad():
            losses = model.loss(data)
            current_loss = 0
            if model_name == 'CM_Flow':
                for loss_element in losses:
                    current_loss += loss_element.mean()
            else:
                current_loss = losses.mean()
        if 'test_loss' in test_dict:
            test_dict["test_loss"].append(current_loss.item())  # add current iter loss to test loss list.
        else:
            test_dict["test_loss"] = [current_loss.item()]  # add current iter loss to test loss list.

        pbar.update(data.size(0))
        pbar.set_description('Test, Log likelihood in epoch {}: {:.6f}'.format(epoch, np.mean(test_dict["test_loss"])))

    pbar.close()

    return test_dict


def train_val(current_model, model_name, args, data_loaders, dataset,
              transform_model_1=None, transform_model_2=None, transform_inputs=True,
              cm_flow=True, test_dict={}, disable=False, grid_search=False):
    best_dict_current_model = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0}
    total_losses_current_model = {'train_loss': [], 'val_loss': []}  # initialize a dict to keep the per-epoch metrics

    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        # Initialize dictionary for epoch losses
        current_epoch_losses = {"train_loss": [], "val_loss": []}

        # Train model
        current_epoch_losses = train(args=args,
                                     epoch=epoch,
                                     model=current_model,
                                     train_loader=data_loaders['train_loader'],
                                     current_epoch_losses=current_epoch_losses,
                                     device=args.device,
                                     model_name=model_name,
                                     transform_model_1=transform_model_1,
                                     transform_model_2=transform_model_2,
                                     transform_inputs=transform_inputs,
                                     disable=disable)

        # Perform Validation
        current_epoch_losses, best_dict_current_model = validate(epoch=epoch,
                                                                 model=current_model,
                                                                 loader=data_loaders['valid_loader'],
                                                                 device=args.device,
                                                                 current_epoch_losses=current_epoch_losses,
                                                                 best_dict=best_dict_current_model,
                                                                 model_name=model_name,
                                                                 transform_model_1=transform_model_1,
                                                                 transform_model_2=transform_model_2,
                                                                 transform_inputs=transform_inputs,
                                                                 disable=disable)

        # Set model state to epoch
        current_model.state['model_epoch'] = epoch

        if not grid_search:
            # save model and best val idx and best val acc, using the model dir, model name and model idx
            save_model(model=current_model,
                       model_save_dir=args.experiment_saved_models,
                       model_save_name="train_current_model", model_idx=epoch,
                       best_validation_model_idx=best_dict_current_model['best_validation_epoch'],
                       best_validation_model_loss=best_dict_current_model['best_validation_loss'])

            # Save mean of each epoch in total losses dictionary
            for key, value in current_epoch_losses.items():
                total_losses_current_model[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            # Save current epoch statistics
            save_statistics(experiment_log_dir=args.experiment_logs, filename='summary_{}.csv'.format(model_name),
                            stats_dict=total_losses_current_model, current_epoch=epoch,
                            continue_from_mode=epoch)  # save statistics to stats file.

        # Early stopping
        if args.early_stopping is True:
            if epoch - best_dict_current_model['best_validation_epoch'] >= 10:
                break

        if not grid_search:
            print('Best validation at epoch {}: Average Log Likelihood: {:.4f}'.
                  format(best_dict_current_model['best_validation_epoch'], best_dict_current_model['best_validation_loss']))

            # @Todo: integrate this
            # Save sample plots every n epochs
            # if model_name == 'RealNVP':
            #     if epoch % args.plot_frequ == 0:
            #         save_samples_plot_copula(args=args,
            #                                  epoch=epoch,
            #                                  model=current_model,
            #                                  dataset=dataset)

    if not grid_search:

        current_model = model_loader(current_model, args.experiment_saved_models, 'train_current_model', epoch, name='')

        # Perform test evaluation
        test_dict = test(epoch=best_dict_current_model['best_validation_epoch'],
                         model=current_model,
                         loader=data_loaders['test_loader'],
                         device=args.device,
                         test_dict=test_dict,
                         model_name=model_name,
                         transform_model_1=transform_model_1,
                         transform_model_2=transform_model_2,
                         transform_inputs=transform_inputs,
                         disable=disable)

        num_samples = int(0.2 * args.obs)

        # Calculate Jensen-Shannon Divergence of copula
        if model_name == 'RealNVP':
            test_dict = jsd_eval_copula(args,
                                        best_dict_current_model['best_validation_epoch'],
                                        current_model,
                                        data_loaders['test_loader'],
                                        args.device,
                                        test_dict=test_dict,
                                        transform_model_1=transform_model_1,
                                        transform_model_2=transform_model_2,
                                        transform_inputs=transform_inputs,
                                        cm_flow=cm_flow)
            # Evaluate copula margins on test set
            test_dict = margin_uniformity(best_dict_current_model['best_validation_epoch'],
                                          current_model,
                                          data_loaders['test_loader'],
                                          args.device,
                                          transform_fct=args.transform_fct,
                                          test_dict=test_dict,
                                          num_samples=num_samples,
                                          cm_flow=args.RealNVP_part_of_CM_Flow)

        if model_name == 'DDSF':
            # Calculate Jensen-Shannon Divergence of marginal 1
            args.marginal = args.marginal
            test_dict = jsd_eval_marginal(marginal=args.marginal,
                                          args=args,
                                          epoch=best_dict_current_model['best_validation_epoch'],
                                          model=current_model,
                                          test_dict=test_dict)

        if model_name == 'DDSF_1':
            # Calculate Jensen-Shannon Divergence of marginal 1
            args.marginal = args.marginal_1
            test_dict = jsd_eval_marginal(marginal=args.marginal_1,
                                          args=args,
                                          epoch=best_dict_current_model['best_validation_epoch'],
                                          model=current_model,
                                          test_dict=test_dict,
                                          plotname='jsd_pretrain_marginal_0')

        # Calculate Jensen-Shannon Divergence of marginal 1
        if model_name == 'DDSF_2':
            args.marginal = args.marginal_2
            test_dict = jsd_eval_marginal(marginal=args.marginal_2,
                                          args=args,
                                          epoch=best_dict_current_model['best_validation_epoch'],
                                          model=current_model,
                                          test_dict=test_dict,
                                          plotname='jsd_pretrain_marginal_1')

        if model_name == 'CM_Flow':
            args.marginal = args.marginal_1
            test_dict = jsd_eval_copula(args,
                                        best_dict_current_model['best_validation_epoch'],
                                        current_model,
                                        data_loaders['test_loader'],
                                        args.device,
                                        test_dict=test_dict,
                                        transform_model_1=transform_model_1,
                                        transform_model_2=transform_model_2,
                                        transform_inputs=transform_inputs,
                                        cm_flow=cm_flow)
            test_dict = jsd_eval_marginal(marginal=args.marginal_1,
                                          args=args,
                                          epoch=best_dict_current_model['best_validation_epoch'],
                                          model=current_model,
                                          test_dict=test_dict,
                                          cm_flow=0,
                                          plotname='jsd_cm_flow_marginal')
            args.marginal = args.marginal_2
            test_dict = jsd_eval_marginal(marginal=args.marginal_2,
                                          args=args,
                                          epoch=best_dict_current_model['best_validation_epoch'],
                                          model=current_model,
                                          test_dict=test_dict,
                                          cm_flow=1,
                                          plotname='jsd_cm_flow_marginal')
            # Evaluate copula margins on test set
            test_dict = margin_uniformity(best_dict_current_model['best_validation_epoch'],
                                          current_model,
                                          data_loaders['test_loader'],
                                          args.device,
                                          transform_fct=args.transform_fct,
                                          test_dict=test_dict,
                                          num_samples=num_samples,
                                          cm_flow=True)

        # Plot losses
        result_dict = collect_experiment_dicts(target_dir=args.experiment_logs, model_name=model_name)
        if model_name == 'DDSF':
            plot_result_graphs(args.figures_path, args.exp_name, args.marginal, result_dict, current_model_name=model_name)
        elif model_name == 'RealNVP':
            plot_result_graphs(args.figures_path, args.exp_name, args.copula, result_dict, current_model_name=model_name)
        else:
            plot_result_graphs(args.figures_path, args.exp_name, args.copula, result_dict, current_model_name=model_name)

    return current_model, best_dict_current_model, test_dict
