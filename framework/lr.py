import numpy as np
import statsmodels.api as sm

from framework.symbol import *
from framework.stats_basic import *
from framework.base import *

# def lr(y):
#     X = np.arange(y.shape[0])
#     X = sm.add_constant(X)
#     model = sm.OLS(y, X).fit()
#     pred = model.predict(X)
#     pred = name(pd.Series(pred, y.index), y.name + " fit")
#     return pred
    
def lr(y, X=None, print_r2=False, ret_coef=False):
    if X is None:
        X = np.arange(len(y))
        if is_series(y):
            xs = y.index
        else:
            xs = X
        name_ = get_name(y) + " fit"
    else:
        xs = X
        name_ = "fit"
        
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if print_r2:
        print(f"R^2: {model.rsquared:.2f}")
    if ret_coef:
        return model.params

    pred = model.predict(X)
    pred = name(pd.Series(pred, xs), name_)
    pred = pred.sort_index()
    return pred
    
def lr_predict(y, X):
    params = lr(*sync(y, X), ret_coef=True, print_r2=True)
    print(params)
    return params.iloc[0] + params.iloc[1] * X

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
