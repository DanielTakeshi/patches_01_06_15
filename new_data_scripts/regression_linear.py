"""
(c) January 2016 by Daniel Seita

This will run ridge regression using python's sklearn library.
"""

import numpy as np
from sklearn import linear_model


def do_regression(data, alpha=0.1):
    """
    Perform regression using ridge regression, and *returns* the predictions.

    :alpha: The parameter used to determine the weight of the coefficients, to prevent overfitting.
    """
    
    print("\nNow on ridge regression with alpha = {}.".format(alpha))
    X_train,y_train = data[0]
    X_val,y_val = data[1]
    regressor = linear_model.Ridge(alpha=alpha)

    # Fit model, use clipping to ensure output in [0,1], and evaluate. Return 'predictions'.
    predictions = regressor.fit(X_train, y_train).predict(X_val) 
    mse = np.mean( (np.clip(predictions,0,1) - y_val) ** 2 )
    mae = np.mean( np.absolute(np.clip(predictions,0,1) - y_val) )
    print("M.S.E. = {:.5f}".format(mse))
    print("M.A.E. = {:.5f}".format(mae))

    # Before returning our actual predictions, first analyze by discretizing the data. Need indices first.
    thresh1 = np.percentile(y_val, 33)
    thresh2 = np.percentile(y_val, 66)
    indices1 = np.where(y_val < thresh1)[0]
    tmp1 = np.where(y_val >= thresh1)[0]
    tmp2 = np.where(y_val < thresh2)[0]
    indices2 = np.intersect1d(tmp1,tmp2)
    indices3 = np.where(y_val >= thresh2)[0]

    # Using indices, extract the appropriate values, clip, take absolute value, then print.
    predictions1 = predictions[indices1]
    y_val1 = y_val[indices1]
    results1 = np.mean(np.absolute(np.clip(predictions1,0,1)-y_val1))
    print("For bottom third, avg abs diff = {:.5f}.".format(results1))
    
    predictions2 = predictions[indices2]
    y_val2 = y_val[indices2]
    results2 = np.mean(np.absolute(np.clip(predictions2,0,1)-y_val2))
    print("For middle third, avg abs diff = {:.5f}.".format(results2))
    
    predictions3 = predictions[indices3]
    y_val3 = y_val[indices3]
    results3 = np.mean(np.absolute(np.clip(predictions3,0,1)-y_val3))
    print("For top third, avg abs diff = {:.5f}.".format(results3))

    # Whew. I'll need to put the above in another method but this will do for now. Return predictions.
    return predictions


if __name__ == '__main__':
    print("We should not be calling this!")
