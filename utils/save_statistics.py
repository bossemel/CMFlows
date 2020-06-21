import os
import csv
import torch


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch,
                    continue_from_mode=False, save_full_dict=False, test_epoch=None):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            current_list = ['epoch']
            current_list.extend(list(stats_dict.keys()))
            writer.writerow(current_list)
        if test_epoch is not None:
            current_list = [test_epoch]
        else:
            current_list = [current_epoch]
        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                current_list.extend(row_to_add)
                writer.writerow(current_list)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            current_list.extend(row_to_add)
            writer.writerow(current_list)

    return summary_filename


def load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(value)

    return stats


def save_model(model, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
               best_validation_model_loss, model_RealNVP=None, model_DDSF_1=None, model_DDSF_2=None):
    """
    Save the network parameter state and current best val epoch idx and best val accuracy.
    :param model_save_name: Name to use to save model without the epoch index
    :param model_idx: The index to save the model with.
    :param best_validation_model_idx: The index of the best validation model to be stored for future use.
    :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
    :param model_save_dir: The directory to store the state at.
    :param state: The dictionary containing the system state.
    """
    def model_saver(model_type, name):
        model_type.state['network'] = model_type.state_dict()  # save network parameter and other variables.
        model_type.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        model_type.state['best_val_model_acc'] = best_validation_model_loss  # save current best val loss
        torch.save(model_type.state, f=os.path.join(model_save_dir, "{}_{}_model{}".format(model_save_name, str(
            model_idx), name)))  # save state at prespecified filepath

    model_saver(model, '')

    if model_RealNVP is not None:
        model_saver(model_RealNVP, '_RealNVP')

    if model_DDSF_1 is not None:
        model_saver(model_DDSF_1, '_DDSF_1')

    if model_RealNVP is not None:
        model_saver(model_DDSF_2, '_DDSF_2')


def model_loader(model_type, model_save_dir, model_save_name, model_idx, name=''):
    state = torch.load(f=os.path.join(model_save_dir, "{}_{}_model{}".format(model_save_name, str(model_idx), name)))
    model_type.load_state_dict(state_dict=state['network'])
    return model_type


def load_model(model, model_save_dir, model_save_name, model_idx,
               model_RealNVP=None, model_DDSF_1=None, model_DDSF_2=None):
    """
    Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
    :param model_save_dir: The directory to store the state at.
    :param model_save_name: Name to use to save model without the epoch index
    :param model_idx: The index to save the model with.
    :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
    """

    def model_loader(model_type, name):
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}_model{}".format(model_save_name, str(model_idx), name)))
        model_type.load_state_dict(state_dict=state['network'])
        return model_type

    model_loader(model, '')

    if model_RealNVP is not None:
        model_loader(model_RealNVP, '_RealNVP')

    if model_DDSF_1 is not None:
        model_loader(model_DDSF_1, '_DDSF_1')

    if model_RealNVP is not None:
        model_loader(model_DDSF_2, '_DDSF_2')
