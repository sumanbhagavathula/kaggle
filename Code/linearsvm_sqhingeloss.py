import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import math

import algorithms as alg


def computeobj(betas, x, y, lambduh):
    # example usage for computeobj:
    # betas=np.repeat(0,spam_data.shape[1])
    # lambduh=0.1
    # computeobj(betas,spam_data,spam_labels,lambduh)

    n = x.shape[0]
    xb = np.dot(x, betas)
    yxb = 1 - y * xb
    zeroindexes = np.where(yxb < 0)
    yxb[zeroindexes] = 0
    # display(yxb.shape)
    summaxsqr = np.sum(yxb ** 2)

    result = (1 * summaxsqr) / n + lambduh * (np.linalg.norm(betas) ** 2)

    return (result)


def computegrad(betas, x, y, lambduh):
    # example usage for computegrad:
    # betas=np.repeat(0,spam_data.shape[1])
    # lambduh=0.1
    # computegrad(betas,spam_data,spam_labels,lambduh)
    n = x.shape[0]
    xb = np.dot(x, betas)
    yxb = 1 - y * xb
    zeroindexes = np.where(yxb < 0)
    yxb[zeroindexes] = 0
    yxt = y * x.T
    summaxprod = np.sum(yxt * yxb, 1)

    result = -2 * summaxprod / n + 2 * lambduh * betas
    return result

def linearSVM(x, y, lambduh, max_iter=1000, method='fastgradientdescent'):
    beta_init = np.zeros(x.shape[1])
    if method == 'fastgradientdescent':
        linearSVMbeta = alg.fastgradientdescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)
    else:
        linearSVMbeta = alg.graddescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)

    return linearSVMbeta


def linearsvmscore(x,y,betas):
    svmout=np.dot(x,betas)
    predict = lambda x: 1 if  x>0 else -1

    failed=0
    for i in range(0,x.shape[0]):
        if(predict(svmout[i]) * y[i] <0):
           failed+=1
    failscore=failed*100.0/x.shape[0]
    return failscore


def linearSVM_CV(x, y, lambda_vals=[], folds=10, max_iter=1000, seed=0, lambda_logbase=10,Num_lambdas=10):
    if len(lambda_vals) == 0:
        lambda_start = 2-Num_lambdas
        lambda_end = 2
        lambda_vals = np.logspace(lambda_start,lambda_end,Num_lambdas, base = lambda_logbase)

    best_lambda_CV = alg.crossvalidation(x, y, lambda_vals, computescore=linearsvmscore, computegrad = computegrad, computeobj = computeobj, folds=folds, max_iter = max_iter, seed=seed)
    return best_lambda_CV

def linearSVM_parallelCV(x, y, lambda_vals=[], folds=10, max_iter=1000, seed=0, lambda_logbase=10,Num_lambdas=10, threads=4):
    if len(lambda_vals) == 0:
        lambda_start = 2-Num_lambdas
        lambda_end = 2
        lambda_vals = np.logspace(lambda_start,lambda_end,Num_lambdas, base = lambda_logbase)

    best_lambda_CV = alg.parallelcrossvalidation(x, y, lambda_vals, computescore=linearsvmscore, computegrad = computegrad, computeobj = computeobj, folds=folds, max_iter = max_iter, seed=seed, threads=4)
    return best_lambda_CV


