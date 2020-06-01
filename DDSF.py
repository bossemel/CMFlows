import datasets.distributions as distributions
from DDSF_modules.DensityEstimator import DensityEstimator
import DDSF_modules.visualizer as visualizer

if __name__ == '__main__':
    res = 200
    rng = [(-5, 5), (-5, 5)]
    distr_1 = distributions.SwissRoll(0.5)
    distr_2 = distributions.Gaussian(0.5)
    denaf = DensityEstimator()
    denaf.fit(distr_2, 2000)
    fig = visualizer.visualize2D(distr_2, denaf, res=res, rng=rng)
    fig.savefig('figures/swissroll_DDSF.png', format='png')
