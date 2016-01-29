"""
(c) January 2016 by Daniel Seita

This will run SVM kernel regression using python's sklearn library.
"""

import numpy as np
from sklearn import svm


def do_regression(data, kernel='linear', cache_size=1000):
    """
    Perform regression using SVM kernel regression.

    :alpha: The parameter used to determine the weight of the coefficients, to prevent overfitting.
    """

    print("\nNow on SVM regression with kernel = {}.".format(kernel))
    X_train,y_train = data[0]
    X_val,y_val = data[1]
    regressor = svm.SVR(kernel=kernel, cache_size=cache_size)

    # Fit model, use clipping to ensure output in [0,1], and evaluate.
    predictions = regressor.fit(X_train, y_train).predict(X_val) 
    mse = np.mean( (np.clip(predictions,0,1) - y_val) ** 2 )
    mae = np.mean( np.absolute(np.clip(predictions,0,1) - y_val) )
    print("M.S.E. = {:.5f}".format(mse))
    print("M.A.E. = {:.5f}".format(mae))

    # TODO later, try to analyze by discretizing the data.


if __name__ == '__main__':
    print("We should not be calling this!")
