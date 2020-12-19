import torch
import torch.utils.data

from tqdm import tqdm
import numpy as np

from utils.load_and_save import save_statistics, save_model, load_model
from utils.loss_plots import collect_experiment_dicts, plot_result_graphs
from NSF_modules.eval import jsd_eval as jsd_eval_copula, margin_uniformity
from NSF_modules.eval import jsd_eval_1D as jsd_eval_marginal #@Todo: replace with NSF module eval 1d
from NSF_modules.visualizer import visualize1D
from CM_modules.visualizer import visualize1D_CM
from CM_modules.utils import jsd_eval_marginal_cm

eps = 0.0001


def single_model_forward(args, model, model_name, data, transform_inputs, device, cond_data=None):
    # When training just the marg_flow or cop_flow, there is only one loss and no preprocessing of data.
    if model_name == 'marg_flow_1':
        loss = model.marg_flow_1.loss(data[:, 0: 1])
    elif model_name == 'marg_flow_2':
        loss = model.marg_flow_2.loss(data[:, 1: 2])
    elif model_name == 'marg_flow_3':
        loss = model.marg_flow_3.loss(data[:, 1: 2])
    elif model_name == 'marg_flow_4':
        loss = model.marg_flow_4.loss(data[:, 1: 2])
    elif model_name == 'cop_flow':
        if transform_inputs is True:
            with torch.no_grad():
                output_marg_flow_1 = model.marg_flow_1.transform_to_noise(data[:, 0: 1]).reshape(-1, 1)
                output_marg_flow_2 = model.marg_flow_2.transform_to_noise(data[:, 1: 2]).reshape(-1, 1)
            if not args.conditional_copula:
                outputs_marg_flows = torch.cat((output_marg_flow_1, output_marg_flow_2), dim=1)
                loss = model.cop_flow.loss(outputs_marg_flows)
            else:
                loss = model.cop_flow.loss(output_marg_flow_1, cond_inputs=output_marg_flow_2)
        else:
            if args.conditional_copula:
                loss = model.cop_flow.loss(inputs=data[:, 0: 1], cond_inputs=data[:, 1: 2])
            else:
                loss = model.cop_flow.loss(data)
    elif model_name == 'rvine_cop_flow':
        loss = model.loss(inputs=data[:, 0: 1], cond_inputs=data[:, 1: 2])
    else:
        loss = model.loss(data)

    return model, loss


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

    pbar = tqdm(total=len(train_loader.dataset), disable=disable_tqdm)

    for batch_idx, data in enumerate(train_loader):
        model.train()

        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        args.optimizer.zero_grad()

        assert not torch.isnan(torch.sum(data))
        model, loss = single_model_forward(args,
                                           model,
                                           model_name,
                                           data,
                                           transform_inputs,
                                           device)
        assert not torch.isnan(loss)
        if 'train_loss' in current_epoch_losses:
            current_epoch_losses["train_loss"].append(loss.item())  # add current iter loss to the train loss list
        else:
            current_epoch_losses['train_loss'] = [loss.item()]

        loss.backward()

        # Perform gradient clipping
        if args.clip_grad_norm:
            if model_name == 'marg_flow_1':
                torch.nn.utils.clip_grad_norm_(model.marg_flow_1.parameters(), args.clip)
            elif model_name == 'marg_flow_2':
                torch.nn.utils.clip_grad_norm_(model.marg_flow_2.parameters(), args.clip)
            elif model_name == 'marg_flow_3':
                torch.nn.utils.clip_grad_norm_(model.marg_flow_2.parameters(), args.clip)
            elif model_name == 'marg_flow_4':
                torch.nn.utils.clip_grad_norm_(model.marg_flow_2.parameters(), args.clip)
            elif model_name == 'cop_flow':
                torch.nn.utils.clip_grad_norm_(model.cop_flow.parameters(), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if args.scheduler is None:
            args.optimizer.step()
        else:
            args.optimizer.step()
            args.scheduler.step()

        pbar.update(data.size(0))
        pbar.set_description('{} Train, Log likelihood: {:.6f}'.format(model_name, loss))

    pbar.close()

    # for module in model.modules():
    #     if isinstance(module, fnn.BatchNormFlow):
    #         module.momentum = 0

    # with torch.no_grad():
    #     if model_name == 'cop_flow':
    #         if transform_inputs is True:
    #             move_transformed_outputs(args, train_loader, device, model, data)
    #         else:
    #             if args.conditional_copula:
    #                 model.forward(inputs=train_loader.dataset.tensors[0][:, 0: 1].to(data.device),
    #                               cond_inputs=train_loader.dataset.tensors[0][:, 1: 2].to(data.device))
    #             else:
    #                 model(train_loader.dataset.tensors[0].to(data.device))
    #     elif model_name == 'CM_Flow':
    #         move_transformed_outputs(args, train_loader, device, model, data)

    # for module in model.modules():
    #     if isinstance(module, fnn.BatchNormFlow):
    #         module.momentum = 1

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

    pbar = tqdm(total=len(loader.dataset), disable=disable_tqdm)
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        model.eval()

        if isinstance(data, list):
            data = data[0]

        data = data.to(device)

        with torch.no_grad():
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
              rvine=False, save_name=None, cm_flow=False):
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

        # save model and best val idx and best val acc, using the model dir, model name and model idx
        save_model(model=model,
                   model_save_dir=args.experiment_saved_models,
                   model_save_name="train_model", model_idx=epoch,
                   best_validation_model_idx=best_dict['best_validation_epoch'],
                   best_validation_model_loss=best_dict['best_validation_loss'])

        if not grid_search and not rvine:
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

    # Load model with best validation epoch
    model = load_model(model, args.experiment_saved_models, 'train_model',
                       best_dict['best_validation_epoch'])

    #
    model.eval()

    if not grid_search and not rvine:

        # # Load model with best validation epoch
        # model = load_model(model, args.experiment_saved_models, 'train_model',
        #                    best_dict['best_validation_epoch'])

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
        if model_name == 'cop_flow':
            test_dict = jsd_eval_copula(args,
                                        best_dict['best_validation_epoch'],
                                        model.cop_flow if cm_flow else model,
                                        data_loaders['test_loader'],
                                        args.device,
                                        test_dict=test_dict,
                                        cm_flow=args.cop_flow_part_of_CM_Flow)
            # Evaluate copula margins on test set
            test_dict = margin_uniformity(args=args,
                                          epoch=best_dict['best_validation_epoch'],
                                          model=model.cop_flow if cm_flow else model,
                                          transform_fct=args.transform_fct,
                                          test_dict=test_dict,
                                          num_samples=num_samples,
                                          cm_flow=args.cop_flow_part_of_CM_Flow)

        if model_name == 'marg_flow':
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

        if model_name == 'marg_flow_1':
            # Calculate Jensen-Shannon Divergence of marginal 1
            args.marginal = args.marginal_1
            test_dict = jsd_eval_marginal(marginal=args.marginal_1,
                                          args=args,
                                          model=model.marg_flow_1,
                                          test_dict=test_dict,
                                          plotname='jsd_pretrain_marginal_0',
                                          cm_flow=True,
                                          marginal_num='1')

        # Calculate Jensen-Shannon Divergence of marginal 1
        if model_name == 'marg_flow_2':
            args.marginal = args.marginal_2
            test_dict = jsd_eval_marginal(marginal=args.marginal_2,
                                          args=args,
                                          model=model.marg_flow_2,
                                          test_dict=test_dict,
                                          plotname='jsd_pretrain_marginal_1',
                                          cm_flow=True,
                                          marginal_num='1')

        # if model_name == 'CM_Flow':
        #     # args.marginal = args.marginal_1
        #     test_dict = jsd_eval_copula(args,
        #                                 best_dict['best_validation_epoch'],
        #                                 model,
        #                                 data_loaders['test_loader'],
        #                                 device=args.device,
        #                                 test_dict=test_dict,
        #                                 cm_flow=args.cop_flow_part_of_CM_Flow)

        #     test_dict = jsd_eval_marginal_cm(marginal_1=args.marginal_1,
        #                                      marginal_2=args.marginal_2,
        #                                      args=args,
        #                                      model=model,
        #                                      test_dict=test_dict,
        #                                      plotname='jsd_cm_flow_marginal')
            # args.marginal = args.marginal_2

            # # Evaluate copula margins on test set
            # test_dict = margin_uniformity(args=args,
            #                               epoch=best_dict['best_validation_epoch'],
            #                               model=model,
            #                               cond_inputs=torch.from_numpy(dataset.tst[:, 1:2]) if args.conditional_copula else None,
            #                               transform_fct=args.transform_fct,
            #                               test_dict=test_dict,
            #                               num_samples=num_samples)

            # if not error_bars:
            #     # Visualize the marginals
            #     visualize1D_CM(model=model,
            #                    epoch=best_dict['best_validation_epoch'],
            #                    args=args,
            #                    best_val=True)

        # Plot losses
        result_dict = collect_experiment_dicts(target_dir=args.experiment_logs, model_type=model_name)
        if not error_bars and not grid_search and not rvine:
            if model_name == 'marg_flow' or model_name == 'DDSF':
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
