import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import multiprocessing

import linearsvm_sqhingeloss as lsvm
import utilityfunctions as util

import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import math
import random
import linearsvmmulticlass_sqhingeloss as lsvmmc
seed=0
feature_count = 50
np.random.seed(seed)
sample_size = 250
indv_sample_size = math.floor(sample_size / 5)
w, h = sample_size, feature_count;
x = pd.DataFrame([[0 for a in range(w)] for b in range(h)])

for j in range(0, feature_count):
    # for i in range(0,math.floor(sample_size/3)):
    random_loc = np.random.uniform(0, 50)
    random_scale = np.random.uniform(200, 675)
    lo = 0
    hi = indv_sample_size
    x.iloc[j, lo:hi] = np.array(np.random.normal(random_loc, random_scale, indv_sample_size))

    random_lo = np.random.normal(120, 180)
    random_hi = np.random.normal(1500, 2100)
    lo = indv_sample_size
    hi = 2 * indv_sample_size
    x.iloc[j, lo:hi] = np.array(np.random.uniform(random_lo, random_hi, indv_sample_size))

    random_loc = np.random.uniform(800, 1100)
    random_scale = np.random.uniform(750, 1200)
    lo = 2 * indv_sample_size
    hi = 3 * indv_sample_size
    x.iloc[j, lo:hi] = np.array(np.random.normal(random_loc, random_scale, indv_sample_size))

    random_loc = np.random.uniform(800, 1100)
    random_scale = np.random.uniform(750, 1200)
    lo = 3 * indv_sample_size
    hi = 4 * indv_sample_size
    x.iloc[j, lo:hi] = np.array(np.random.normal(random_loc, random_scale, indv_sample_size))

    random_loc = np.random.uniform(800, 1100)
    random_scale = np.random.uniform(750, 1200)
    lo = 4 * indv_sample_size
    hi = 5 * 2 * indv_sample_size
    x.iloc[j, lo:hi] = np.array(np.random.normal(random_loc, random_scale, indv_sample_size))

data = np.transpose(x)

labels = pd.DataFrame(np.concatenate(
    [np.repeat(1, indv_sample_size), np.repeat(2, indv_sample_size), np.repeat(3, indv_sample_size),
     np.repeat(4, indv_sample_size), np.repeat(5, indv_sample_size)]))

x=np.asanyarray(data)
y=np.array(labels.iloc[:,0])

x_train=copy.deepcopy(x)
y_train=copy.deepcopy(y)
x_test=copy.deepcopy(x)
y_test=copy.deepcopy(y)

threads=multiprocessing.cpu_count()
folds=10

dirname = 'sklearnlinearsvm'
path=util.makeresultsdir(dirname)

#mylinearsvmCV_lambda = lsvm.linearSVM_parallelCV(x_train, y_train, lambda_vals, folds=folds, max_iter=1000, threads=threads)
mylinearsvmCV_lambda, cvscores = lsvm.linearSVM_parallelCV(x_train, y_train, folds=folds, max_iter=1000, threads=threads)
print("Best lambda found from cross validation: " + str(mylinearsvmCV_lambda))

dirname = 'linearsvm_sqhingeloss'
path=util.makeresultsdir(dirname)

#save validation results
display(type(cvscores))
display(cvscores.shape)
display(cvscores.iloc[:,0])
display(cvscores.iloc[:,1])

util.plotcrossvalidationresults(cvscores.iloc[:,0],cvscores.iloc[:,1], logbase = 10, pathtosave=path)

lsvm_betas = lsvm.linearSVM(x_train, y_train, mylinearsvmCV_lambda, max_iter=1000, method='fastgradientdescent')

print('fitted coefficients: ')
display(lsvm_betas)

validationscore = lsvm.linearsvmscore(x_test,y_test,lsvm_betas)

print('Validation score on test-set = ' + str(validationscore))

text_file = open(path+'/results.txt', "w")
text_file.write('Validation score on test-set = ' + str(validationscore))
text_file.close()

#save results
lsvm_betas.to_csv(path+'/betas.csv',header=False,index=False)

print("done!!!")
