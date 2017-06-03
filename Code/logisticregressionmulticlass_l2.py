import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import math
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process, Array
import logisticregression_l2regularization as logregl2
import algorithms as alg
import utilityfunctions as util

def logisticregressionmulticlass(fitclasslabel, computeobj, computegrad, x, y, lambduh, max_iter=1000,
                                     method='fastgradientdescent', seed=0):

    uniqueclasses = np.unique(y).astype(int)
    classindex = [x[0] for x in np.where(uniqueclasses == fitclasslabel)]
    ysub = y - fitclasslabel
    classidxs = (ysub == 0)
    notclassidxs = np.logical_not(classidxs)
    x1 = x[classidxs]
    xrest = x[notclassidxs]
    fraction = xrest.shape[0] / x1.shape[0]
    while fraction <= 1:
        xrest = np.append(x1, xrest, axis=0)
        fraction = xrest.shape[0] / x1.shape[0]
    np.random.seed(seed)
    np.random.shuffle(xrest)
    xrest = xrest[0:x1.shape[0], :]

    x = np.append(x1, xrest, axis=0)

    y1 = np.repeat(1, x1.shape[0])
    yrest = np.repeat(-1, x1.shape[0])

    y = np.append(y1, yrest, axis=0)

    beta_init = np.zeros(x.shape[1])
    if method == 'fastgradientdescent':
        logregmcl2beta = alg.fastgradientdescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)
    else:
        logregmcl2beta = alg.graddescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)

    return logregmcl2beta

def logisticregmulticlasswrapper(args):
    return logisticregressionmulticlass(*args)

def logisticregmulticlass_l2(x, y, lambduh, max_iter=1000, method='fastgradientdescent', seed=0, threads=4):

    # implements one vs rest strategy
    # get the number of unique classes from the training labels
    uniqueclasses = np.unique(y).astype(int)
    print(uniqueclasses)
    numclasses = uniqueclasses.size
    numfeatures = x.shape[1]


    logisticregmcbetas = np.zeros((numclasses, numfeatures))

    pool = ThreadPool(threads)
    params = [(fitclasslabel, logregl2.computeobj, logregl2.computegrad, x, y, lambduh, max_iter, 'fastgradientdescent', seed) for fitclasslabel in uniqueclasses]

    logisticregmcbetas = pool.map(logisticregmulticlasswrapper, params)

    pool.close()
    # pool.join()

    return np.asarray(logisticregmcbetas)

def logisticregmulticlass_predict(betas, x_new, classlabels):
    multiclasslogodds = (np.dot(x_new, betas.T))
    multiclassprobs = np.exp(multiclasslogodds) / (1 + np.exp(multiclasslogodds))

    predclassidxs = np.argmax(multiclassprobs, axis=1)

    return classlabels[predclassidxs]


#def predictmulticlass(betas,x_new,availableclasslabels):
#    multiclasslogodds = (np.dot(x_new, betas.T))
#    multiclassprobs = np.exp(multiclasslogodds) / (1 + np.exp(multiclasslogodds))

#    predclassidxs = np.argmax(multiclassprobs, axis=1)

#    return availableclasslabels[predclassidxs]

