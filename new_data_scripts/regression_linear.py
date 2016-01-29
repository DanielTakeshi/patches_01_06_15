"""
(c) January 2016 by Daniel Seita

This will run ridge regression using python's sklearn library.
"""

import numpy as np
from sklearn import linear_model


def do_regression(data, alpha=0.1):
    """
    Perform regression using ridge regression.

    :alpha: The parameter used to determine the weight of the coefficients, to prevent overfitting.
    """
    
    print("\nNow on ridge regression with alpha = {}.".format(alpha))
    X_train,y_train = data[0]
    X_val,y_val = data[1]
    regressor = linear_model.Ridge(alpha=alpha)

    # Fit model, use clipping to ensure output in [0,1], and evaluate.
    predictions = regressor.fit(X_train, y_train).predict(X_val) 
    mse = np.mean( (np.clip(predictions,0,1) - y_val) ** 2 )
    mae = np.mean( np.absolute(np.clip(predictions,0,1) - y_val) )
    print("M.S.E. = {:.5f}".format(mse))
    print("M.A.E. = {:.5f}".format(mae))

    # TODO later, try to analyze by discretizing the data.


if __name__ == '__main__':
    print("We should not be calling this!")
