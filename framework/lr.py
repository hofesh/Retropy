import numpy as np
import statsmodels.api as sm

from framework.symbol import *
from framework.stats_basic import *
from framework.base import *

def lr(y):
    X = np.arange(y.shape[0])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    pred = name(pd.Series(pred, y.index), y.name + " fit")
    return pred
    
def lr_beta(y, X=None, pvalue=False):
    if X is None:
        X = np.arange(y.shape[0])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        if pvalue:
            return model.params[1], model.pvalues[1]
        return model.params[1]
    else:
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        if pvalue:
            return model.params[1], model.pvalues[1]
        return model.params[1]
    
def lrret_beta(y, X, freq=None, pvalue=False):
    y = get(y, freq=freq)
    X = get(X, freq=freq)
    y = logret(y)
    X = logret(X)
    y, X = sync(y, X)
    return lr_beta(y, X, pvalue=pvalue)
