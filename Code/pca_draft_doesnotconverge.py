import numpy as np
import random
import pandas as pd
import math
from IPython.core.display import display
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import sys

import utilityfunctions as util

def deflate(Z, a):
    return Z - Z.dot(np.outer(a, a))


def getunitnormvector(length):
    a_0 = np.random.standard_normal(length)
    return a_0 / np.linalg.norm(a_0,axis=0)

def display_results(v, lambdas):
    plt.plot(lambdas)
    plt.xlabel('Pass through data set')
    # Note that the top eigenvalue is the negative of the value of the objective function.
    plt.ylabel('Estimated value of top eigenvalue')
    plt.show()
    print('Estimated top eigenvalue:', lambdas[-1])
    #print('Estimated corresponding eigenvector:', v)
    print(v)


def oja(Z, a_0, eta_0, t_0, num_epochs):
    t = 5
    a = a_0
    eta = eta_0
    oja_a = np.arange(0, 10 * Z.shape[1], 1.0).reshape(10, Z.shape[1])
    top_eig_value = np.zeros(10)
    for iter in range(0, num_epochs):
        Z = np.random.permutation(Z)
        for i in range(0, Z.shape[0]):
            eta = eta_0 / (t + t_0)
            a = a + eta * np.dot(Z[i, :], np.dot(np.transpose(Z[i, :]), a))
            a = a / np.linalg.norm(a)
            t = t + 1

        # print(a)

        if (iter >= num_epochs - 10):
            oja_a[10 - (num_epochs - iter), :] = copy.deepcopy(a)
            top_eig_value[10 - (num_epochs - iter)] = a.dot(Z.T).dot(Z).dot(a) / Z.shape[0]

    return np.mean(oja_a, axis=0), np.mean(top_eig_value)


def getempiricalcovariance(x,centered=True):
    n, d = features = x.shape

    if centered != True:
        xbar = np.repeat(np.array(np.sum(x, axis=1) / n), d).reshape(n, d)
        x = x - xbar

    cov = (np.dot(x, x.T)) / n

    return np.cov(x.T)


# function to calculate explained variance ratio
def calcexplvarratio(d, eigvals):
    n = eigvals.size
    cov = getempiricalcovariance(d)
    totalvariance = sum(cov.diagonal())
    # print(totalvariance)
    oja_expl_var_ratio = np.repeat(0., n)
    for i in range(0, n, 1):
        # print(sum(eigvals[0:i+1]))
        oja_expl_var_ratio[i] = sum(eigvals[0:i + 1]) / totalvariance

    return oja_expl_var_ratio


def chooseinitialparams(Z, pcompiter, eta_0, a_0, numepochs, sampleratio=0.25, seed=0):
    isconverged = False
    t_0 = 1
    i = 0
    np.random.seed(seed)
    eigvals=[]
    while isconverged != True:
        eta0_prev = eta_0
        # t_0 = np.random.randint(1,10)
        sampleindexes = np.random.randint(0, Z.shape[1], round(sampleratio * Z.shape[0]))
        #Zsmall = np.asarray(pd.DataFrame(Z).iloc[sampleindexes])
        Zsmall=copy.deepcopy(Z)
        a_initial, eigval = oja(copy.deepcopy(Zsmall), a_0, eta_0, t_0, numepochs)
        eigvals = np.append(eigvals, eigval)

        if util.isconverged(eigvals):
            print('converged in iteration ' + str(i))
            isconverged = True

        i += 1

        #if i % 100 == 0:
        print("iteration number:", str(i))

    eta_0 /= sampleratio  # since we are searching using a subset of the data
    print("found eta_0", eta_0)

    return eta_0, t_0


def ojaexplainedvariancetarget(data, targetvariance, numepochs):
    a_0 = getunitnormvector(data.shape[1])
    eta_0 = 10 ** (-4)
    t_0 = 1
    #Z = data - np.mean(data, axis=0)
    Z=copy.deepcopy(data)
    cov = 1 / Z.shape[0] * np.dot(Z, Z.T)
    la_eigvals, la_eigvecs = np.linalg.eig(cov)

    pcompiter = 0
    a_oja = a_0
    eigvals = []
    principalcomponents = []
    # explainedvariance = 0
    explainedvariance = []
    explainedvariance = np.append(explainedvariance, 0.0)
    while explainedvariance[-1] < targetvariance:
        print("Search for principal component number: ", pcompiter + 1)
        if pcompiter != 0:
            Z = deflate(Z, a_oja)
        eta_0, t_0 = chooseinitialparams(copy.deepcopy(Z), pcompiter, eta_0 / 10, a_0, numepochs)
        a_oja, eigval = oja(copy.deepcopy(Z), a_0, eta_0, t_0, round(numepochs * (1 + pcompiter / 5)))
        principalcomponents = np.append(principalcomponents, a_oja)
        eigvals = np.append(eigvals, eigval)
        explainedvariance = np.append(explainedvariance, calcexplvarratio(data, eigvals)[-1])
        print("explained variance so far with ", pcompiter + 1, " pricipal components = ", explainedvariance[-1])
        print("eigen value = ", eigval)
        print("eigen value from linalg", np.sort(la_eigvals)[-pcompiter - 1])

        pcompiter = pcompiter + 1

        if pcompiter >= data.shape[1]:
            pcompiter = -1
            print("All principal components completed.")
            break

    numpcomps = pcompiter
    principalcomponents = principalcomponents.reshape(numpcomps, data.shape[1])

    return numpcomps, eigvals, explainedvariance, principalcomponents