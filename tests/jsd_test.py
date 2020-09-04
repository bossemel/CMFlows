import scipy.stats
from utils import js_divergence


if __name__ == '__main__':
    # What i need: X_in_p, X_in_q, Y_in_p, Y_in_q

    # 1. case: p=q
    mv_gaussian = mvnorm = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., 0],
                                                                             [0, 1.]])
    samples = mv_gaussian.rvs(100000)
    samples_2 = mv_gaussian.rvs(100000)

    X_in_p = mv_gaussian.pdf(samples)
    X_in_q = mv_gaussian.pdf(samples)
    Y_in_p = mv_gaussian.pdf(samples_2)
    Y_in_q = mv_gaussian.pdf(samples_2)

    print('Should be zero:')
    print(js_divergence(X_in_p, X_in_q, X_in_p, X_in_q))
    print('Should be small:')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # 2nd case:
    mv_gaussian_2 = mvnorm = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5],
                                                                               [0.5, 1.]])
    samples_2 = mv_gaussian_2.rvs(100000)

    X_in_p = mv_gaussian.pdf(samples)
    X_in_q = mv_gaussian_2.pdf(samples)
    Y_in_p = mv_gaussian.pdf(samples_2)
    Y_in_q = mv_gaussian_2.pdf(samples_2)

    print('Should be bigger:')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))

    # 3rd case:
    mv_gaussian_2 = mvnorm = scipy.stats.multivariate_normal(mean=[10000, 10000], cov=[[10000., 0.5],
                                                                                       [0.5, 10000.]])
    samples_2 = mv_gaussian_2.rvs(100000)

    X_in_p = mv_gaussian.pdf(samples)
    X_in_q = mv_gaussian_2.pdf(samples)
    Y_in_p = mv_gaussian.pdf(samples_2)
    Y_in_q = mv_gaussian_2.pdf(samples_2)

    print('Should be even bigger:')
    print(js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q))
