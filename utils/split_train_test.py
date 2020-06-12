from sklearn import model_selection


def split_train_val_test(xx):
    """Splits data into train, val and test set, using 80/20/280 split.

    Params:
        xx: data to split

    Returns:
        train, val, test: train, val and test set
    """
    train, testval = model_selection.train_test_split(xx, test_size=0.2)
    val, test = model_selection.train_test_split(testval, test_size=0.5)
    return train, val, test
