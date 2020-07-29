import scipy.stats
import numpy as np
import torch
import math

def flow_density(inputs, log_jacob):
    """Calculate density using change of variable formula.

    Params:
        inputs: data array
        log_jacob: accumulated log jacobian determinant

    Returns:
        log density array
    """
    log_prob = (-0.5 * inputs.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    return (log_prob + log_jacob).sum(-1, keepdim=True)


def log_normal(x, log_jacob, log_var, eps=0.00001):
    c = torch.tensor(- 0.5 * np.log(2 * np.pi))
    return - x ** 2 / (2. * log_var + eps) - log_var / 2. + c + log_jacob


if __name__ == '__main__':
    obs = 1000
    gaussian_1 = scipy.stats.norm(loc=2, scale=2)
    gaussian_2 = scipy.stats.norm(loc=12, scale=2)

    samples_1 = gaussian_1.rvs(int(obs)).reshape(-1, 1)
    samples_2 = gaussian_2.rvs(int(obs)).reshape(-1, 1)

    pdf_1 = gaussian_1.pdf(samples_1)
    pdf_2 = gaussian_2.pdf(samples_2)
    n=obs
    zeros = torch.FloatTensor(n, 1).zero_()

    pred_density_ddsf = log_normal(torch.tensor(pdf_1), torch.tensor(pdf_2), zeros + 1.0).sum(1)
    pred_density_realnvp = flow_density(torch.tensor(pdf_1), torch.tensor(pdf_2))

    print(pred_density_ddsf.mean())
    print(pred_density_realnvp.mean())
    print(pred_density_realnvp.mean() - 0.5)
