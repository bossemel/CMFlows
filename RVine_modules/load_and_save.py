import os
import torch


def save_rvine(experiment_log_dir, filename, rvine_object):
    rvine_filename = os.path.join(experiment_log_dir, filename)
    rvine_filename_trees = rvine_filename + 'tree_list'
    torch.save(rvine_object.tree_list, rvine_filename_trees)


def load_rvine(experiment_log_dir, filename, rvine_object):
    rvine_filename = os.path.join(experiment_log_dir, filename)
    rvine_filename_trees = rvine_filename + 'tree_list'
    tree_list = torch.load(rvine_filename_trees)
    rvine_object.tree_list = tree_list
