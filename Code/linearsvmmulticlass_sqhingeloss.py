import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import math
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process, Array

import linearsvm_sqhingeloss as lsvm
import algorithms as alg
import load_and_clean_data as lcd
import utilityfunctions as util

def linearSVMmulticlass(fitclasslabel, computeobj, computegrad, x, y, lambduh, max_iter=1000, method='fastgradientdescent', seed=0):
    uniqueclasses = np.unique(y).astype(int)
    classindex=[x[0] for x in np.where(uniqueclasses==fitclasslabel)]
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
        linearsvmbeta = alg.fastgradientdescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)
    else:
        linearsvmbeta = alg.graddescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=max_iter)

    return linearsvmbeta

def linearSVMmulticlasswrapper(args):
    return linearSVMmulticlass(*args)

#get best lambda from cross validation and use it here
def linearsvmmulticlass_sqhingeloss(x, y, lambduh, max_iter=1000, method='fastgradientdescent', seed=0, threads=4):
    #implements one vs rest strategy
    #get the number of unique classes from the training labels
    uniqueclasses = np.unique(y).astype(int)
    print(uniqueclasses)
    numclasses = uniqueclasses.size
    numfeatures = x.shape[1]

    linearsvmmulticlassbetas = np.zeros((numclasses,numfeatures))

    #run 2-class linear SVM and stack results, in parallel for n-1 cases
    pool = ThreadPool(threads)
    params=[(fitclasslabel, lsvm.computeobj, lsvm.computegrad, x, y, lambduh, max_iter, 'fastgradientdescent', seed) for fitclasslabel  in uniqueclasses]

    linearsvmmulticlassbetas = pool.map(linearSVMmulticlasswrapper, params)

    pool.close()
    #pool.join()

    return np.asarray(linearsvmmulticlassbetas)

def linearsvmmulticlass_predict(betas,x_new,classlabels):
    multiclasslogodds = (np.dot(x_new, betas.T))
    multiclassprobs = np.exp(multiclasslogodds) / (1 + np.exp(multiclasslogodds))

    predclassidxs = np.argmax(multiclassprobs, axis=1)

    return classlabels[predclassidxs]


def predictmulticlass(betas,x_new,availableclasslabels):
    multiclasslogodds = (np.dot(x_new, betas.T))
    multiclassprobs = np.exp(multiclasslogodds) / (1 + np.exp(multiclasslogodds))

    predclassidxs = np.argmax(multiclassprobs, axis=1)

    return availableclasslabels[predclassidxs]







