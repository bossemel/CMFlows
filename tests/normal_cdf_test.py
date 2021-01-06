import torch
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    obs = 1000

    # plt.hist(noise)
    # plt.show()

    noise = torch.Tensor(obs, 2).normal_()
    fig = plt.figure()
    fig = sns.jointplot(noise[:, 0:1], noise[:, 1:2], kind='hex', stat_func=None)
    fig.savefig('test', dpi=300, bbox_inches='tight')

    normal_distr = torch.distributions.normal.Normal(0, 1)
    noise[:, 1:2] = noise[:, 1:2] / 2
    noise_uni = normal_distr.cdf(noise)

    fig = plt.figure()
    fig = sns.jointplot(noise_uni[:, 0:1], noise_uni[:, 1:2], kind='hex', stat_func=None)
    fig.savefig('test_uni', dpi=300, bbox_inches='tight')
