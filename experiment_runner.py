import torch
import torch.utils.data
from tqdm import tqdm

from utils.load_and_save import save_statistics, save_model, load_model
from utils.loss_plots import collect_experiment_dicts, plot_result_graphs
from NSF_modules.eval import jsd_eval as jsd_eval_copula, margin_uniformity
from NSF_modules.eval import jsd_eval_1d as jsd_eval_marginal
from NSF_modules.visualizer import visualize1d

eps = 1e-07


def single_model_forward(model, model_name, data, conditional):
    # When training just the marg_flow or cop_flow, there is only one loss and no preprocessing of data.
    if 'marg_flow_1' in model_name:
        loss = model.loss(data[:, 0:1])
    elif 'marg_flow_2' in model_name:
        loss = model.loss(data[:, 1:2])
    elif model_name in ['cop_flow', 'cop_flow_rv', 'cop_flow_real']:
        if conditional:
            loss = model.loss(inputs=data[:, 0:1], context=data[:, 1:2])
        else:
            loss = model.loss(data)
    else:
        loss = model.loss(data)
    return model, loss


def train(args, model, train_loader, current_epoch_losses, device,
          model_name=None, disable_tqdm=False):
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
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        args.optimizer.zero_grad()
        model, loss = single_model_forward(model,
                                           model_name,
                                           data,
                                           args.conditional_copula)
        assert not torch.isnan(loss)
        if 'train_loss' in current_epoch_losses:
            current_epoch_losses["train_loss"].append(loss.detach())  # add current iter loss to the train loss list
        else:
            current_epoch_losses['train_loss'] = [loss.detach()]

        loss.backward()

        # Perform gradient clipping
        if args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        args.optimizer.step()
        args.scheduler.step()

        pbar.update(data.size(0))
        pbar.set_description('{} Train, Log likelihood: {:.6f}'.format(model_name, loss))

    pbar.close()

    return current_epoch_losses


def validate(args, epoch, model, loader, device,
             current_epoch_losses=None, best_dict=None, model_name=None,
             disable_tqdm=False):
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

        if isinstance(data, list):
            data = data[0]

        data = data.to(device)

        with torch.no_grad():
            model, loss = single_model_forward(model,
                                               model_name,
                                               data,
                                               args.conditional_copula)

        if 'val_loss' in current_epoch_losses:
            current_epoch_losses["val_loss"].append(loss.detach())  # add current iter loss to val loss list.
        else:
            current_epoch_losses["val_loss"] = [loss.detach()]  # add current iter loss to val loss list.

        pbar.update(data.size(0))

    val_mean_loss = torch.mean(torch.tensor(current_epoch_losses['val_loss']))
    pbar.set_description('{} Val, Log likelihood: {:.6f}'.format(model_name, val_mean_loss))

    if val_mean_loss < best_dict['best_validation_loss']:
        best_dict['best_validation_loss'] = val_mean_loss
        best_dict['best_validation_epoch'] = epoch

    pbar.close()
    return current_epoch_losses, best_dict


def test(args, epoch, model, loader, device,
         test_dict, model_name, disable_tqdm=False):
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
    pbar = tqdm(total=len(loader.dataset), disable=disable_tqdm)
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            model, loss = single_model_forward(model,
                                               model_name,
                                               data,
                                               args.conditional_copula)
        test_loss_name = model_name + '_test_loss'
        if 'test_loss' in test_dict:
            test_dict[test_loss_name].append(loss.detach())  # add current iter loss to test loss list.
        else:
            test_dict[test_loss_name] = [loss.detach()]  # add current iter loss to test loss list.

        pbar.update(data.size(0))
    pbar.set_description('Test, Log likelihood in epoch {}: {:.6f}'.format(epoch,
                                                                           torch.mean(torch.tensor(
                                                                               test_dict[test_loss_name]))))
    pbar.close()

    return test_dict


def train_val(model, model_name, args, data_loaders,
              disable_tqdm=False, hp_search=False, error_bars=False,
              rvine=False, save_name=None):
    best_dict = {'best_validation_loss': float('inf'), 'best_validation_epoch': 0}
    total_losses = {'train_loss': [], 'val_loss': []}  # initialize a dict to keep the per-epoch metrics
    test_dict = {}

    for epoch in range(args.epochs):
        if not hp_search and not error_bars and not rvine:
            print('\nEpoch: {}'.format(epoch))

        # Initialize dictionary for epoch losses
        current_epoch_losses = {"train_loss": [], "val_loss": []}

        # Train model
        model.train()
        current_epoch_losses = train(args=args,
                                     model=model,
                                     train_loader=data_loaders['train_loader'],
                                     current_epoch_losses=current_epoch_losses,
                                     device=args.device,
                                     model_name=model_name,
                                     disable_tqdm=disable_tqdm)

        # Perform Validation
        model.eval()
        current_epoch_losses, best_dict = validate(args=args,
                                                   epoch=epoch,
                                                   model=model,
                                                   loader=data_loaders['valid_loader'],
                                                   device=args.device,
                                                   current_epoch_losses=current_epoch_losses,
                                                   best_dict=best_dict,
                                                   model_name=model_name,
                                                   disable_tqdm=disable_tqdm)

        # Set model state to epoch
        model.state['model_epoch'] = epoch

        # save model and best val idx and best val acc, using the model dir, model name and model idx
        save_model(model=model,
                   model_save_dir=args.experiment_saved_models,
                   model_save_name="train_model", model_idx=epoch,
                   best_validation_model_idx=best_dict['best_validation_epoch'],
                   best_validation_model_loss=best_dict['best_validation_loss'])

        if not hp_search and not rvine:
            # Save mean of each epoch in total losses dictionary
            for key, value in current_epoch_losses.items():
                total_losses[key].append(torch.mean(
                    torch.tensor(value)).item())

            # Save current epoch statistics
            save_statistics(experiment_log_dir=args.experiment_logs, filename='summary_{}.csv'.format(model_name),
                            stats_dict=total_losses, current_epoch=epoch,
                            continue_from_mode=epoch)  # save statistics to stats file.

        if not hp_search and not rvine:
            print('Best validation at epoch {}: Average Log Likelihood: {:.4f}'.
                  format(best_dict['best_validation_epoch'], best_dict['best_validation_loss']))

    # Load model with best validation epoch
    model = load_model(model, args.experiment_saved_models, 'train_model',
                       best_dict['best_validation_epoch'])

    # Save best model under different name
    save_model(model=model,
               model_save_dir=args.experiment_saved_models,
               model_save_name="best_epoch_model", model_idx=best_dict['best_validation_epoch'],
               best_validation_model_idx=best_dict['best_validation_epoch'],
               best_validation_model_loss=best_dict['best_validation_loss'],
               save_name=save_name)

    model.eval()

    if not hp_search and not rvine:
        # Plot losses
        result_dict = collect_experiment_dicts(target_dir=args.experiment_logs, model_type=model_name)
        if not error_bars and not hp_search and not rvine:
            plot_result_graphs(args.figures_path, args.exp_name,
                               args.marginal if model_name in ['marg_flow', 'marg_flow_1', 'marg_flow_2']
                               else args.copula,
                               result_dict, model_type=model_name)

        # # Perform test evaluation
        test_dict = test(args=args,
                         epoch=best_dict['best_validation_epoch'],
                         model=model,
                         loader=data_loaders['test_loader'],
                         device=args.device,
                         test_dict=test_dict,
                         model_name=model_name,
                         disable_tqdm=disable_tqdm)

        # Calculate Jensen-Shannon Divergence of copula
        if model_name == 'cop_flow':
            test_dict = jsd_eval_copula(args=args,
                                        epoch=best_dict['best_validation_epoch'],
                                        model=model,
                                        test_dict=test_dict)
            # Evaluate copula margins on test set
            test_dict = margin_uniformity(args=args,
                                          epoch=best_dict['best_validation_epoch'],
                                          model=model,
                                          test_dict=test_dict,
                                          num_samples=args.obs,
                                          cm_flow=args.cop_flow_part_of_CM_Flow)

        if model_name == 'marg_flow':
            # Calculate Jensen-Shannon Divergence of marginal 1
            args.marginal = args.marginal
            test_dict = jsd_eval_marginal(args=args,
                                          model=model,
                                          test_dict=test_dict)

        # if model_name == 'marg_flow_1':
        #     # Calculate Jensen-Shannon Divergence of marginal 1
        #     args.marginal = args.marginal_1
        #     test_dict = jsd_eval_marginal(marginal=args.marginal_1,
        #                                   args=args,
        #                                   model=model,
        #                                   test_dict=test_dict,
        #                                   plotname='jsd_pretrain_marginal_0',
        #                                   cm_flow=True,
        #                                   marginal_num='1')

        # # Calculate Jensen-Shannon Divergence of marginal 1
        # if model_name == 'marg_flow_2':
        #     args.marginal = args.marginal_2
        #     test_dict = jsd_eval_marginal(marginal=args.marginal_2,
        #                                   args=args,
        #                                   model=model,
        #                                   test_dict=test_dict,
        #                                   plotname='jsd_pretrain_marginal_1',
        #                                   cm_flow=True,
        #                                   marginal_num='1')

        if 'marg_flow_1' in model_name and not error_bars:
            args.marginal = args.marginal_1
            visualize1d(model=model,
                        epoch=best_dict['best_validation_epoch'],
                        args=args,
                        best_val=True,
                        name=model_name)

        if 'marg_flow_2' in model_name and not error_bars:
            args.marginal = args.marginal_2
            visualize1d(model=model,
                        epoch=best_dict['best_validation_epoch'],
                        args=args,
                        best_val=True,
                        name=model_name)
    if rvine:
        if 'marg_flow_rv' in model_name and not error_bars:
            visualize1d(model=model,
                        epoch=best_dict['best_validation_epoch'],
                        args=args,
                        best_val=True,
                        name=model_name + save_name)

    return best_dict, test_dict
