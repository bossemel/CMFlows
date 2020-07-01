import scipy.stats
import numpy as np

if __name__ == '__main__':
    obs = 1000
    gaussian_1 = scipy.stats.norm(loc=2, scale=2)
    gaussian_2 = scipy.stats.norm(loc=12, scale=2)\

    samples_1 = gaussian_1.rvs(int(obs / 2)).reshape(-1, 1)
    samples_2 = gaussian_2.rvs(int(obs / 2)).reshape(-1, 1)

    bimodal_gaussian = np.concatenate([samples_1, samples_2], axis=1)
