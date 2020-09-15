import torch
import torch.utils.data

from tqdm import tqdm
import numpy as np

import RealNVP_modules.flows as fnn

from utils.load_and_save import save_statistics, save_model, load_model
from utils.loss_plots import collect_experiment_dicts, plot_result_graphs
from RealNVP_modules.eval import jsd_eval as jsd_eval_copula, margin_uniformity
from DDSF_modules.eval import jsd_eval as jsd_eval_marginal
from DDSF_modules.visualizer import visualize1D
from CM_modules.visualizer import visualize1D_CM
from CM_modules.utils import jsd_eval_marginal_cm
from utils import flow_density, empty_logdets_context

eps = 0.0001


def ddsf_1_forward(model, data, device):
    logdets, context = empty_logdets_context(data, device)
    return model.model_DDSF_1.forward((data[:, 0: 1], logdets, context))


def ddsf_2_forward(model, data, device):
    logdets, context = empty_logdets_context(data, device)
    return model.model_DDSF_2.forward((data[:, 1:2], logdets, context))


def cm_flow_forward(model, data, device):
    # CM_Flow passes each dimension of the data through a DDSF, and then passes the output through the RealNVP
    output_DDSF_1, logdets_DDSF_1, __ = ddsf_1_forward(model, data, device)
    output_DDSF_2, logdets_DDSF_2, __ = ddsf_2_forward(model, data, device)

    with torch.no_grad():
        outputs_DDSFs = torch.cat((output_DDSF_1, output_DDSF_2), dim=1)

    output_RealNVP, logdets_RealNVP = model.model_RealNVP.forward(inputs=outputs_DDSFs, mode='direct')

    # Calculate losses using Change of Variable Theorem and a normal prior
    loss_DDSF_1 = -flow_density(output_DDSF_1, logdets_DDSF_1.reshape(-1, 1)).mean()
    loss_DDSF_2 = -flow_density(output_DDSF_2, logdets_DDSF_2.reshape(-1, 1)).mean()
    loss_RealNVP = -flow_density(output_RealNVP, logdets_RealNVP).mean()
    assert loss_RealNVP >= 0
    assert loss_DDSF_1 >= 0
    assert loss_DDSF_2 >= 0
    loss = loss_RealNVP + loss_DDSF_1 + loss_DDSF_2
    return model, loss, loss_DDSF_1, loss_DDSF_2, loss_RealNVP


def single_model_forward(args, model, model_name, data, transform_inputs, device):
    # When training just the DDSF or RealNVP, there is only one loss and no preprocessing of data.
    if model_name == 'DDSF_1':
        output_DDSF_1, logdets_DDSF_1, __ = ddsf_1_forward(model, data, device)
        loss = -flow_density(output_DDSF_1, logdets_DDSF_1.reshape(-1, 1)).mean()
    elif model_name == 'DDSF_2':
        output_DDSF_2, logdets_DDSF_2, __ = ddsf_2_forward(model, data, device)
        loss = -flow_density(output_DDSF_2, logdets_DDSF_2.reshape(-1, 1)).mean()
    elif model_name == 'RealNVP':
        if transform_inputs is True:
            with torch.no_grad():
                output_DDSF_1, logdets_DDSF_1, __ = ddsf_1_forward(model, data, device)
                output_DDSF_2, logdets_DDSF_2, __ = ddsf_2_forward(model, data, device)
                outputs_DDSFs = torch.cat((output_DDSF_1, output_DDSF_2), dim=1)
            if args.conditional_copula:
                output_RealNVP, logdets_RealNVP = model.model_RealNVP(inputs=output_DDSF_1, cond_inputs=output_DDSF_2, mode='direct')
            else:
                output_RealNVP, logdets_RealNVP = model.model_RealNVP(inputs=outputs_DDSFs, mode='direct')
            loss = -flow_density(output_RealNVP, logdets_RealNVP).mean()
        else:
            if args.conditional_copula:
                losses = model.loss(inputs=data[:, 0: 1], cond_inputs=data[:, 1: 2])
            else:
                losses = model.loss(data)
            loss = losses.mean()
    else:
        losses = model.loss(data)
        loss = losses.mean()
        del losses

    return model, loss


def move_transformed_outputs(args, train_loader, device, model, data):
    logdets, context = empty_logdets_context(train_loader.dataset.tensors[0], device)
    output_DDSF_1, logdets_DDSF_1, __ = ddsf_1_forward(model, train_loader.dataset.tensors[0], device)
    output_DDSF_2, logdets_DDSF_2, __ = ddsf_2_forward(model, train_loader.dataset.tensors[0], device)
    if args.conditional_copula:
        model.model_RealNVP.forward(inputs=output_DDSF_1.to(data.device), cond_inputs=output_DDSF_2.to(data.device))
    else:
        train_loader_tensor = torch.cat((output_DDSF_1, output_DDSF_2), dim=1)
        model.model_RealNVP.forward(train_loader_tensor.to(data.device))


def train(args, epoch, model, train_loader, current_epoch_losses, device,
          model_name=None,
          transform_inputs=True, disable_tqdm=False):
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

    pbar = tqdm(total=len(train_loader.dataset), disable=disable_tqdm)

    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        assert not torch.isnan(torch.sum(data))

        if model_name == 'CM_Flow':
            model, loss, loss_DDSF_1, loss_DDSF_2, loss_RealNVP = cm_flow_forward(model,
                                                                                  data,
                                                                                  device)
            # Append Training loss to current_epoch_losses dictionary
            if 'train_loss' in current_epoch_losses:
                current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list
            else:
                current_epoch_losses['train_loss'] = []

            # Perform backwards calculation for each loss on its own, retaining the first two graphs. Since the outputs
            # of the DDSF's are inputs to RealNVP, the RealNVP loss contains the DDSF weights as well.
            loss_DDSF_1.backward(retain_graph=True)
            loss_DDSF_2.backward(retain_graph=True)
            loss_RealNVP.backward()

            if args.clip_grad_norm:
                model.clip_grad_norm()

        else:
            if model_name in ['DDSF_1', 'DDSF_2', 'RealNVP']:
                model, loss = single_model_forward(args,
                                                   model,
                                                   model_name,
                                                   data,
                                                   transform_inputs,
                                                   device)

            else:
                losses = model.loss(data)
                loss = losses.mean()

            if 'train_loss' in current_epoch_losses:
                current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list
            else:
                current_epoch_losses['train_loss'] = [loss.item()]

            loss.backward()

        # Perform gradient clipping
        if model_name in ['DDSF_1', 'DDSF_2', 'DDSF', 'CM_Flow']:
            if args.clip_grad_norm:
                model.clip_grad_norm()

        # Perform one optimizer step
        args.optimizer.step()
        args.optimizer.zero_grad()

        pbar.update(data.size(0))
        pbar.set_description('{} Train, Log likelihood: {:.6f}'.format(model_name, loss))

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        if model_name == 'RealNVP':
            if transform_inputs is True:
                move_transformed_outputs(args, train_loader, device, model, data)
            else:
                if args.conditional_copula:
                    model(inputs=train_loader.dataset.tensors[0][:, 0: 1].to(data.device),
                          cond_inputs=train_loader.dataset.tensors[0][:, 1: 2].to(data.device))
                else:
                    model(train_loader.dataset.tensors[0].to(data.device))
        elif model_name == 'CM_Flow':
            move_transformed_outputs(args, train_loader, device, model, data)

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

    return current_epoch_losses


def validate(args, epoch, model, loader, device,
             current_epoch_losses=None, best_dict=None, model_name=None,
             transform_model_1=None, transform_model_2=None, transform_inputs=True, disable_tqdm=False):
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

    pbar = tqdm(total=len(loader.dataset), disable=disable_tqdm)
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]

        data = data.to(device)

        with torch.no_grad():
            if model_name == 'CM_Flow':
                model, loss, loss_DDSF_1, loss_DDSF_2, loss_RealNVP = cm_flow_forward(model,
                                                                                      data,
                                                                                      device)
            else:
                model, loss = single_model_forward(args,
                                                   model,
                                                   model_name,
                                                   data,
                                                   transform_inputs,
                                                   device)

        if 'val_loss' in current_epoch_losses:
            current_epoch_losses["val_loss"].append(loss.item())  # add current iter loss to val loss list.
        else:
            current_epoch_losses["val_loss"] = [loss.item()]  # add current iter loss to val loss list.

        val_mean_loss = np.mean(current_epoch_losses['val_loss'])
        if val_mean_loss < best_dict['best_validation_loss']:  # if current epoch's mean val acc is greater than the saved best val acc then
            best_dict['best_validation_loss'] = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
            best_dict['best_validation_epoch'] = epoch  # set the experiment-wise best val idx to be the current epoch's idx

        pbar.update(data.size(0))
        pbar.set_description('{} Val, Log likelihood: {:.6f}'.format(model_name, val_mean_loss))

    pbar.close()
    return current_epoch_losses, best_dict


def test(args, epoch, model, loader, device,
         test_dict, model_name,
         transform_inputs=True,
         disable_tqdm=False):
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

    pbar = tqdm(total=len(loader.dataset), disable=disable_tqdm)
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]

        data = data.to(device)
        with torch.no_grad():
            if model_name == 'CM_Flow':
                model, loss, loss_DDSF_1, loss_DDSF_2, loss_RealNVP = cm_flow_forward(model,
                                                                                      data,
                                                                                      device)
            else:
                model, loss = single_model_forward(args,
                                                   model,
                                                   model_name,
                                                   data,
                                                   transform_inputs,
                                                   device)
        test_loss_name = model_name + '_test_loss'
        if 'test_loss' in test_dict:
            test_dict[test_loss_name].append(loss.item())  # add current iter loss to test loss list.
        else:
            test_dict[test_loss_name] = [loss.item()]  # add current iter loss to test loss list.

        pbar.update(data.size(0))
        pbar.set_description('Test, Log likelihood in epoch {}: {:.6f}'.format(epoch, np.mean(test_dict[test_loss_name])))

    pbar.close()

    return test_dict


def train_val(model, model_name, args, data_loaders, dataset,
              transform_model_1=None, transform_model_2=None, transform_inputs=True,
              test_dict={}, disable_tqdm=False, grid_search=False, error_bars=False,
              rvine=False, save_name=None):
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0}
    total_losses = {'train_loss': [], 'val_loss': []}  # initialize a dict to keep the per-epoch metrics

    for epoch in range(args.epochs):
        if not grid_search and not error_bars and not rvine:
            print('\nEpoch: {}'.format(epoch))

        # Initialize dictionary for epoch losses
        current_epoch_losses = {"train_loss": [], "val_loss": []}

        # Train model
        current_epoch_losses = train(args=args,
                                     epoch=epoch,
                                     model=model,
                                     train_loader=data_loaders['train_loader'],
                                     current_epoch_losses=current_epoch_losses,
                                     device=args.device,
                                     model_name=model_name,
                                     transform_inputs=transform_inputs,
                                     disable_tqdm=disable_tqdm)

        # Perform Validation
        current_epoch_losses, best_dict = validate(args=args,
                                                   epoch=epoch,
                                                   model=model,
                                                   loader=data_loaders['valid_loader'],
                                                   device=args.device,
                                                   current_epoch_losses=current_epoch_losses,
                                                   best_dict=best_dict,
                                                   model_name=model_name,
                                                   transform_inputs=transform_inputs,
                                                   disable_tqdm=disable_tqdm)

        # Set model state to epoch
        model.state['model_epoch'] = epoch

        if not grid_search and not rvine:
            # save model and best val idx and best val acc, using the model dir, model name and model idx
            save_model(model=model,
                       model_save_dir=args.experiment_saved_models,
                       model_save_name="train_model", model_idx=epoch,
                       best_validation_model_idx=best_dict['best_validation_epoch'],
                       best_validation_model_loss=best_dict['best_validation_loss'])

            # Save mean of each epoch in total losses dictionary
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            # Save current epoch statistics
            save_statistics(experiment_log_dir=args.experiment_logs, filename='summary_{}.csv'.format(model_name),
                            stats_dict=total_losses, current_epoch=epoch,
                            continue_from_mode=epoch)  # save statistics to stats file.

        # Early stopping
        if args.early_stopping:
            if epoch - best_dict['best_validation_epoch'] >= 10:
                break

        if not grid_search and not rvine:
            print('Best validation at epoch {}: Average Log Likelihood: {:.4f}'.
                  format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

    if not grid_search and not rvine:

        # Load model with best validation epoch
        model = load_model(model, args.experiment_saved_models, 'train_model',
                           best_dict['best_validation_epoch'])

        # Perform test evaluation
        test_dict = test(args=args,
                         epoch=best_dict['best_validation_epoch'],
                         model=model,
                         loader=data_loaders['test_loader'],
                         device=args.device,
                         test_dict=test_dict,
                         model_name=model_name,
                         transform_inputs=transform_inputs,
                         disable_tqdm=disable_tqdm)

        num_samples = int(0.2 * args.obs)

        # Calculate Jensen-Shannon Divergence of copula
        if model_name == 'RealNVP':
            test_dict = jsd_eval_copula(args,
                                        best_dict['best_validation_epoch'],
                                        model,
                                        data_loaders['test_loader'],
                                        args.device,
                                        test_dict=test_dict,
                                        cm_flow=args.RealNVP_part_of_CM_Flow)
            # Evaluate copula margins on test set
            test_dict = margin_uniformity(args=args,
                                          epoch=best_dict['best_validation_epoch'],
                                          model=model,
                                          cond_inputs=torch.from_numpy(dataset.tst[:, 1: 2]),
                                          transform_fct=args.transform_fct,
                                          test_dict=test_dict,
                                          num_samples=num_samples,
                                          cm_flow=args.RealNVP_part_of_CM_Flow)

        if model_name == 'DDSF':
            # Calculate Jensen-Shannon Divergence of marginal 1
            args.marginal = args.marginal
            test_dict = jsd_eval_marginal(marginal=args.marginal,
                                          args=args,
                                          model=model,
                                          test_dict=test_dict)

            if not error_bars:
                # Visualize the marginals
                visualize1D(model=model,
                            epoch=best_dict['best_validation_epoch'],
                            args=args,
                            best_val=True)

        if model_name == 'DDSF_1':
            # Calculate Jensen-Shannon Divergence of marginal 1
            args.marginal = args.marginal_1
            test_dict = jsd_eval_marginal(marginal=args.marginal_1,
                                          args=args,
                                          model=model,
                                          test_dict=test_dict,
                                          plotname='jsd_pretrain_marginal_0',
                                          cm_flow=True,
                                          marginal_num='1')

        # Calculate Jensen-Shannon Divergence of marginal 1
        if model_name == 'DDSF_2':
            args.marginal = args.marginal_2
            test_dict = jsd_eval_marginal(marginal=args.marginal_2,
                                          args=args,
                                          model=model,
                                          test_dict=test_dict,
                                          plotname='jsd_pretrain_marginal_1',
                                          cm_flow=True,
                                          marginal_num='1')

        if model_name == 'CM_Flow':
            # args.marginal = args.marginal_1
            test_dict = jsd_eval_copula(args,
                                        best_dict['best_validation_epoch'],
                                        model,
                                        data_loaders['test_loader'],
                                        device=args.device,
                                        test_dict=test_dict,
                                        cm_flow=args.RealNVP_part_of_CM_Flow)

            test_dict = jsd_eval_marginal_cm(marginal_1=args.marginal_1,
                                             marginal_2=args.marginal_2,
                                             args=args,
                                             model=model,
                                             test_dict=test_dict,
                                             plotname='jsd_cm_flow_marginal')
            # args.marginal = args.marginal_2

            # Evaluate copula margins on test set
            test_dict = margin_uniformity(args=args,
                                          epoch=best_dict['best_validation_epoch'],
                                          model=model,
                                          cond_inputs=torch.from_numpy(dataset.tst[:, 1:2]),
                                          transform_fct=args.transform_fct,
                                          test_dict=test_dict,
                                          num_samples=num_samples)

            if not error_bars:
                # Visualize the marginals
                visualize1D_CM(model=model,
                               epoch=best_dict['best_validation_epoch'],
                               args=args,
                               best_val=True)

        # Plot losses
        result_dict = collect_experiment_dicts(target_dir=args.experiment_logs, model_type=model_name)
        if not error_bars and not grid_search and not rvine:
            if model_name == 'DDSF':
                plot_result_graphs(args.figures_path, args.exp_name, args.marginal, result_dict, model_type=model_name)
            else:
                plot_result_graphs(args.figures_path, args.exp_name, args.copula, result_dict, model_type=model_name)

    # Save best model under different name
    save_model(model=model,
               model_save_dir=args.experiment_saved_models,
               model_save_name="best_epoch_model", model_idx=best_dict['best_validation_epoch'],
               best_validation_model_idx=best_dict['best_validation_epoch'],
               best_validation_model_loss=best_dict['best_validation_loss'],
               save_name=save_name)

    return best_dict, test_dict
