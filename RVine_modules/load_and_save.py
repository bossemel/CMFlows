import os
import torch


def save_rvine(experiment_log_dir, filename, rvine_object):
    rvine_filename = os.path.join(experiment_log_dir, filename)
    rvine_filename_trees = rvine_filename + 'tree_list'
    rvine_num_inputs = rvine_filename + 'num_inputs'
    torch.save(rvine_object.tree_list, rvine_filename_trees)
    torch.save(rvine_object.num_inputs, rvine_num_inputs)


def load_rvine(experiment_log_dir, filename, rvine_object):
    rvine_filename = os.path.join(experiment_log_dir, filename)
    rvine_filename_trees = rvine_filename + 'tree_list'
    rvine_num_inputs = rvine_filename + 'num_inputs'
    tree_list = torch.load(rvine_filename_trees)
    num_inputs = torch.load(rvine_num_inputs)
    rvine_object.tree_list = tree_list
    rvine_object.num_inputs = num_inputs
