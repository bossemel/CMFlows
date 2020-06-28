import seaborn as sns
import matplotlib.pyplot as plt
import os

#@Todo: comments

def visualize_joint(data, args, name):
    fig = plt.figure()
    fig = sns.jointplot(data[:, 0], data[:, 1], kind='hex', stat_func=None)
    fig.set_axis_labels('X1', 'X2', fontsize=16)
    fig.savefig(os.path.join(args.figures_path, name + '.pdf'), dpi=300, bbox_inches='tight')
