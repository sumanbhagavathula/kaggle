import numpy as np
import pandas as pd
from IPython.core.display import display
import copy

import algorithms as alg


def computegrad(betas, x, y, eta):
    yx = y[:, np.newaxis] * x
    denom = 1 + np.exp(-yx.dot(betas))
    grad = 1 / len(y) * np.sum(-yx * np.exp(-yx.dot(betas[:, np.newaxis])) / denom[:, np.newaxis],axis = 0) + 2 * eta * betas

    return (grad)


def computeobj(betas, x, y, eta):
    # example usage for f_logistic:
    # betas=np.repeat(0,spam_data.shape[1])
    # eta=0.1
    # f_logistic(betas,spam_data,spam_labels,eta)

    n = x.shape[0]
    result = 1/len(y) * np.sum(np.log(1 + np.exp(-y*x.dot(betas)))) + eta * np.linalg.norm(betas)**2
    return (result)

def logisticregression(beta_init, x, y, lambduh, initstepsize=1, max_iter=1000, method='fastgradientdescent'):
    if method == 'fastgradientdescent':
        logisticbeta = alg.fastgradientdescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)
    else:
        logisticbeta = alg.graddescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)

    return logisticbeta

def computeerror(x, y, betas, FindYHat):
    findyhat = lambda x: 1 if x > 0.5 else -1
    logityhat = np.exp(np.dot(x, betas))
    pyhat = logityhat / (1 + logityhat)

    yhat = np.repeat(0, len(pyhat))

    for i in range(0, len(pyhat)):
        yhat[i] = findyhat(pyhat[i])

    return (np.average(y != yhat))

def logisticregressionscore(x,y,betas):
    xb=np.dot(x,betas)

    probs=np.exp(xb)/(1+np.exp(xb))

    predict = lambda x: 1 if  x>0.5 else -1

    failed=0
    for i in range(0,x.shape[0]):
        if(predict(probs[i]) * y[i] <0):
           failed+=1
    failscore=failed*100.0/x.shape[0]
    return failscore

def logisticregression_CV(x, y, lambda_vals=[], folds=10, max_iter=1000, seed=0, lambda_logbase=2,Num_lambdas=10):
    if len(lambda_vals) == 0:
        lambda_0 = 2 /x.shape[0] * np.max(np.abs(np.dot(x.T, y)))
        lambda_n = lambda_0 / (2**Num_lambdas)
        lambda_vals = np.logspace(lambda_n, lambda_0, Num_lambdas, base=lambda_logbase)

    best_lambda_CV = alg.crossvalidation(logisticregressionscore, computegrad,computeobj,x,y,lambda_vals, folds=folds, max_iter = max_iter, seed=seed)
    return best_lambda_CV


def logisticregression_parallelCV(x, y, lambda_vals=[], folds=10, max_iter=1000, seed=0, lambda_logbase=10, Num_lambdas=10, threads=4):

    if len(lambda_vals) == 0:
        lambda_start = 2 - Num_lambdas
        lambda_end = 0
        lambda_vals = np.logspace(lambda_start, lambda_end, Num_lambdas, base=lambda_logbase)
        lambda_0 = 2 / x.shape[0] * np.max(np.abs(np.dot(x.T, y)))
        lambda_n = lambda_0 / (2 ** Num_lambdas)
        lambda_vals = np.logspace(lambda_n, lambda_0, Num_lambdas, base=lambda_logbase)

    best_lambda_CV = alg.parallelcrossvalidation(x, y, lambda_vals, computescore=logisticregressionscore, computegrad=computegrad, computeobj=computeobj, folds=folds, max_iter=max_iter, seed=seed, threads=4)

    return best_lambda_CV