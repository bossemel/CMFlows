import datasets.distributions


def load_data(args):
    """Data Loader

    Params:
        args: args passed by Training Options

    Returns:
        dataset: full dataset
        num_inputs: dimensions of data
        data_loaders: dictionary containing train, val and test set loader
    """

    dataset = datasets.distributions.Joint_Distr(args.copula, args.marginal_1, args.marginal_2, args.theta, args.obs, mu=args.mu, var=args.var, alpha=args.alpha, no_val=True)

    return dataset
