import os
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def collect_experiment_dicts(target_dir, test_flag=False):
    experiment_dicts = dict()
    for subdir, dir, files in os.walk(target_dir):
        for file in files:
            filepath = None
            if not test_flag:
                if file == 'summary.csv':
                    filepath = os.path.join(subdir, file)

            elif test_flag:
                if file == 'test_summary.csv':
                    filepath = os.path.join(subdir, file)

            if filepath is not None:

                with open(filepath, 'r') as read_file:
                    lines = read_file.readlines()

                current_experiment_dict = {key: [] for key in lines[0].replace('\n', '').split(',')}
                idx_to_key = {idx: key for idx, key in enumerate(lines[0].replace('\n', '').split(','))}

                for line in lines[1:]:
                    for idx, value in enumerate(line.replace('\n', '').split(',')):
                        current_experiment_dict[idx_to_key[idx]].append(float(value))

                experiment_dicts[subdir.split('/')[-2]] = current_experiment_dict

    return experiment_dicts


def plot_result_graphs(figures_path, model_name, plot_name, stats, notebook=True):

    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['train_loss', 'val_loss']:
        item = stats[model_name][k]
        ax_1.plot(np.arange(0, len(item)),
                  item, label='{}_{}'.format(model_name, k))

    ax_1.legend(loc=0)
    ax_1.set_ylabel('Loss')
    ax_1.set_xlabel('Epoch number')

    path = os.path.join(figures_path, '{}_loss_performance.pdf'.format(plot_name))
    fig_1.savefig(path, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format='pdf',
                  transparent=False, bbox_inches=None, pad_inches=0.1)


if __name__ == '__main__':
    figures_path = 'figures/RealNVP'
    experiment_folder = os.path.abspath('default_name')
    experiment_dir = os.path.abspath(os.path.join(experiment_folder, "result_outputs"))
    # experiment_dir = 'path/to/mlpractical_directory'
    result_dict = collect_experiment_dicts(target_dir=experiment_dir)
    print(result_dict)
    for key, value in result_dict.items():
        print(key, list(value.keys()))
    plot_result_graphs(figures_path, 'default_name', 'default_tryout', result_dict)
