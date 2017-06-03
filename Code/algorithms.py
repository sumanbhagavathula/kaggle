import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
from scipy import linalg
from scipy import stats

from multiprocessing.dummy import Pool as ThreadPool

def graddescent(computeobj, computegrad, betas, x, y, eta, max_iter=1000):
    # example usage:
    # betas=np.repeat(10,spam_data.shape[1])
    # graddescent(betas,spam_data,spam_labels,0.1,1000)

    beta_vals = np.zeros((max_iter, x.shape[1]))
    grad_b = computegrad(betas, x, y, eta)
    iter = 0
    while (iter < max_iter):
        eta = backtracking(computeobj, computegrad, x, y, eta, betas)
        betas = betas - eta * grad_b
        beta_vals[iter] = betas
        grad_b = computegrad(betas, x, y, eta)
        iter = iter + 1

    return (beta_vals)


def fastgradientdescent(computeobj, computegrad, beta_init, x, y, lambduh, max_iter=1000):
    # example usage:
    # betas=np.repeat(10,spam_data.shape[1])
    # mylinearsvm(betas,spam_data,spam_labels,0.1,1000)

    iter = 0
    beta_old = beta_init
    theta = beta_init
    theta_vals = beta_init

    scores = []
    grad_theta = computegrad(theta, x, y, lambduh)

    initstepsize = estimateinitialstepsize(x, lambduh)

    while (iter < max_iter):
        # print("backtrack in iteration: " + str(iter))
        eta = backtracking(computeobj, computegrad, x, y, lambduh, theta, t=initstepsize, alpha=0.5, beta=0.8,
                           max_iter=100)
        betas = theta - eta * grad_theta
        theta = betas + ((iter / (iter + 3)) * (betas - beta_old))
        # print("computegrad in iteration: " + str(iter))
        theta_vals = np.vstack((theta_vals, theta))
        grad_theta = computegrad(theta, x, y, lambduh)
        beta_old = betas
        newscore = computeobj(theta, x, y, lambduh)
        # print(newscore)
        scores = np.append(scores, newscore)
        if isconverged(scores) == True:
            break
        iter = iter + 1

    return (theta)

def backtracking(computeobj, computegrad, x, y, lambduh, betas, t=1, alpha=0.5, beta=0.8, max_iter=100):
    # example usage:
    # betas=np.repeat(0,spam_data.shape[1])
    # backtracking(computeobj,computegrad,spam_data,spam_labels,1,betas)
    grad_b = computegrad(betas, x, y, lambduh)
    norm_grad_b_sqr = np.dot(np.transpose(grad_b), grad_b)

    found_t = 0
    iter = 0
    while (found_t == 0 and iter < max_iter):
        if (computeobj(betas - t * grad_b, x, y, lambduh) < computeobj(betas, x, y, lambduh) - alpha * t * (norm_grad_b_sqr)):
            found_t = 1
            # print("found a t in iter:" + str(iter))
        elif (iter == max_iter):
            # print("backtracking...max iters reached...")
            break
        else:
            t = t * beta
            iter = iter + 1

    return t


# initial step size estimate
def estimateinitialstepsize(x, lambduh, seed=0):
    n = x.shape[0]
    p = x.shape[1]
    largeN = False

    np.random.seed(seed)
    if n > 1000000:
        largeN = True
        x_subsample = x.sample(frac=0.1)

    if largeN:
        print("large N")
        L = (linalg.eigh((np.dot(np.transpose(x_subsample), x_subsample)) / x_subsample.shape[0], eigvals_only=True,
                         eigvals=(p - 1, p - 1))) / lambduh
    else:
        L = (linalg.eigh((np.dot(np.transpose(x), x)) / n, eigvals_only=True, eigvals=(p - 1, p - 1))) / lambduh

    return (1 / L + lambduh)


def isconverged(score):
    isconverged = False

    if len(score) >= 30:
        mean=np.mean(score)
        score=score-mean
        k,p=stats.mstats.normaltest(score)

        if p>=0.05:
            #print('converged after ' + str(len(score)) + ' iterations')
            isconverged = True

    return isconverged

#computeobj, computegrad, beta_init, x, y, lambduh, max_iter=1000
#scoring function
#initial lambdas
def crossvalidation(x, y, lambda_vals, computescore, computeobj, computegrad, folds=10, max_iter=1000, seed=0):
    np.random.seed(seed)
    num_obs = x.shape[0]
    fold_indexes = np.arange(0, num_obs)
    fold_indexes = np.random.permutation(fold_indexes)

    print("Trying with the following lambdas (and " + str(folds) + " folds each): ")
    display(lambda_vals)
    k = len(lambda_vals)
    scores = np.zeros(k)
    foldbetas = np.zeros(x.shape[1])

    for ki in range(0, k):
        subscores = []
        for fold in range(0, folds):
            indexstart = fold * round(num_obs / folds)
            indexend = fold * round(num_obs / folds) + round(num_obs / folds)

            # print("fold: "+ str(fold) + " indexstart: " + str(indexstart) + ", indexend: " + str(indexend))

            x_test = x[indexstart:indexend]
            x_train = np.concatenate((x[0:indexstart], x[indexend:]), axis=0)
            y_test = y[indexstart:indexend]
            y_train = np.concatenate((y[0:indexstart], y[indexend:]), axis=0)

            x_train = np.asarray(pd.DataFrame(x_train).apply(
                lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))
            x_test = np.asarray(pd.DataFrame(x_test).apply(
                lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))

            foldbetas = foldbetas[0:x_train.shape[1]]
            foldbetas = fastgradientdescent(computeobj, computegrad, foldbetas, x_train, y_train, lambda_vals[ki], max_iter=max_iter)
            # display(linearsvmbetas[-1])

            newsubscore = computescore(x_test, y_test, foldbetas)
            subscores = np.append(subscores, newsubscore)
            if isconverged(subscores) == True:
                break

                # print("lambda: " + str(lambda_vals[ki]) + ", fold: " + str(fold) + ", subscore: " + str(subscores[fold]))
        scores[ki] = np.mean(subscores)
        #print("lambda " + str(lambda_vals[ki]) + " has mean score " + str(scores[ki]))

    cvresults = pd.DataFrame(np.vstack((lambda_vals, scores)).T)
    cvresults.columns = ['lambdas','CV scores']

    return lambda_vals[np.argmin(scores)], cvresults


def runcvfolds(lambduh,computescore, computeobj, computegrad, num_obs, x, y, folds, max_iter, seed=0):
    np.random.seed(0)
    try:
        folds
    except NameError:
        folds = 5

    try:
        max_iter
    except NameError:
        max_iter = 1000

    subscores = []
    foldbetas = np.zeros(x.shape[1])

    fold_indexes = np.arange(0, num_obs)
    fold_indexes = np.random.permutation(fold_indexes)

    for fold in np.arange(0, folds):
        indexstart = fold * round(num_obs / folds)
        indexend = fold * round(num_obs / folds) + round(num_obs / folds)

        # print("fold: "+ str(fold) + " indexstart: " + str(indexstart) + ", indexend: " + str(indexend))

        x_test = x[indexstart:indexend]
        x_train = np.concatenate((x[0:indexstart], x[indexend:]), axis=0)
        y_test = y[indexstart:indexend]
        y_train = np.concatenate((y[0:indexstart], y[indexend:]), axis=0)

        x_train = np.asarray(pd.DataFrame(x_train).apply(
            lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))
        x_test = np.asarray(pd.DataFrame(x_test).apply(
            lambda x: (x - np.mean(x)) if (np.std(x) == 0) else (x - np.mean(x)) / np.std(x)))

        foldbetas = foldbetas[0:x_train.shape[1]]
        foldbetas = fastgradientdescent(computeobj, computegrad, foldbetas, x_train, y_train, lambduh, max_iter=max_iter)
        # display(linearsvmbetas[-1])

        newsubscore = computescore(x_test, y_test, foldbetas)
        subscores = np.append(subscores, newsubscore)
        if isconverged(subscores) == True:
            break

        #print("lambda: "+ str(lambduh) + ", fold: " + str(fold) + ", meanscore: " + str(newsubscore))

    #print("lambda: " + str(lambduh) + ", meanscore after " + str(folds) + " folds is: " + str(np.mean(subscores)))

    return np.mean(subscores)

def runcvfoldswrapper(args):
    return runcvfolds(*args)


def parallelcrossvalidation(x, y, lambda_vals, computescore, computeobj, computegrad, folds=10, max_iter=1000, seed=0, threads=4):
    np.random.seed(seed)
    num_obs = x.shape[0]

    print("Trying with the following lambdas (and " + str(folds) + " folds each): ")
    display(lambda_vals)
    k = len(lambda_vals)
    scores = np.zeros(k)

    pool = ThreadPool(threads)
    params=[(lambduh,computescore, computeobj, computegrad, num_obs, x, y, folds, max_iter) for lambduh  in lambda_vals]

    scores = pool.map(runcvfoldswrapper, params)
    cvresults = pd.DataFrame(np.vstack((lambda_vals, scores)).T)
    cvresults.columns = ['lambdas','CV scores']

    pool.close()
    pool.join()

    return lambda_vals[np.argmin(scores)], cvresults





