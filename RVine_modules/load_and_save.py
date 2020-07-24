import pickle
import os
import torch


def save_rvine(experiment_log_dir, filename, rvine_object):
    rvine_filename = os.path.join(experiment_log_dir, filename)
    torch.save(rvine_object, f=rvine_filename)
    # with open(rvine_filename, "wb") as file:
    #     pickle.dump(rvine_object, file, -1)


def load_rvine(experiment_log_dir, filename):
    # @Todo: find out what -1 does
    rvine_filename = os.path.join(experiment_log_dir, filename)
    #return pickle.load(open(rvine_filename, "rb", -1))
    return torch.load(f=rvine_filename)
