import numpy as np
import torch
from RVine_modules.options import TrainOptions
from RVine_modules.model_rvine import RVine
import os
from pathlib import Path
from utils.visualizer import visualize_joint
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
eps = 0.0001

if __name__ == '__main__':
    args = TrainOptions().parse(print=False)   # get training options
    args.exp_name = 'rvine_uniform'
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, args.figures_path)
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

    args.RealNVP_part_of_CM_Flow = True
    args.disable_marginal = True
    # Create Folders
    args.epochs = 10
    args.obs = 10000
    args.disable_marginal = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    theta = 2
    obs = args.obs
    transform_fct = 'gaussian'
    copula = 'clayton'

    distr_target = scipy.stats.uniform()
    samples_target = distr_target.rvs((obs, 3))
    assert not np.isnan(samples_target.sum())
    visualize_joint(samples_target, args.figures_path, '4D_cop_samples')

    rv = RVine(args=args, num_inputs=samples_target.shape[1])
    rv.fit(torch.from_numpy(samples_target))
    rv.plot()

    samples_target_2 = distr_target.rvs((obs, 3))
    pdf = rv.pdf_uniform(inputs=samples_target_2)
    print(pdf[:10])
    print(np.mean(pdf), np.std(pdf))

    xx = torch.linspace(0, 1, 10000).reshape(-1, 1).cpu().numpy()
    xx_2 = distr_target.rvs((obs, 2))

    pdf_xx = rv.pdf_uniform(inputs=np.concatenate([xx, xx_2], axis=1))

    fig = plt.figure(figsize=(8, 6))
    plt.plot(xx, pdf_xx, label='Predicted PDF', color='royalblue', linewidth=3.0)
    fig.savefig(os.path.join(args.figures_path, 'uniform_pdf' + '.pdf'), dpi=300, bbox_inches='tight')

    samples_pred = rv.sample(obs, transform=True).detach()
    assert torch.max(samples_pred) <= 1
    assert torch.min(samples_pred) >= 0
    visualize_joint(torch.cat([samples_pred[:, 0:1], samples_pred[:, 1:2]], axis=1), args.figures_path, '4D_rvine_samples_01')
    visualize_joint(torch.cat([samples_pred[:, 1:2], samples_pred[:, 2:3]], axis=1), args.figures_path, '4D_rvine_samples_12')
    visualize_joint(torch.cat([samples_pred[:, 0:1], samples_pred[:, 2:3]], axis=1), args.figures_path, '4D_rvine_samples_12')
    #visualize_joint(torch.cat([samples_pred[:, 2:3], samples_pred[:, 3:4]], axis=1), args.figures_path, '4D_rvine_samples_23')
    #visualize_joint(torch.cat([samples_pred[:, 1:2], samples_pred[:, 3:4]], axis=1), args.figures_path, '4D_rvine_samples_13')
    #visualize_joint(torch.cat([samples_pred[:, 0:1], samples_pred[:, 3:4]], axis=1), args.figures_path, '4D_rvine_samples_03')
