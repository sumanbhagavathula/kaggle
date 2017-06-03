import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
import multiprocessing

import linearsvm_sqhingeloss as lsvm
import utilityfunctions as util

spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', delim_whitespace=True,header=None)

spam=spam[~np.isnan(spam).any(axis=1)]

spam_labels=np.asarray(spam[57])
spam_labels[spam_labels == 0] = -1

spam_data = np.asarray(spam[spam.columns[0:57]])

test_indicator = np.array(pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest' , sep=' ',header=None)).T[0]

x_train=spam_data[test_indicator==0,]
x_test=spam_data[test_indicator==1,]
y_train=spam_labels[test_indicator==0]
y_test=spam_labels[test_indicator==1]

x_train = np.asarray(pd.DataFrame(x_train).apply(lambda x: (x - np.mean(x)) if(np.std(x) == 0) else (x - np.mean(x))/np.std(x)))
x_test=np.asarray(pd.DataFrame(x_test).apply(lambda x: (x - np.mean(x)) if(np.std(x) == 0) else (x - np.mean(x))/np.std(x)))

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
display(cvscores.iloc[0,:])
display(cvscores.iloc[1,:])

util.plotcrossvalidationresults(cvscores.iloc[1,:],cvscores.iloc[1,:], logbase = 10, pathtosave=path)

lsvm_betas = lsvm.linearSVM(x_train, y_train, mylinearsvmCV_lambda, max_iter=1000, method='fastgradientdescent')

print('fitted coefficients: ')
display(lsvm_betas)

validationscore = lsvm.linearsvmscore(x_test,y_test,lsvm_betas)

print('Validation score on test-set = ' + str(validationscore))

text_file = open(path+'/results.txt', "w")
text_file.write('Validation score on test-set = ' + str(validationscore))
text_file.close()

#save results
pd.DataFrame(lsvm_betas).to_csv(path+'/betas.csv',header=False,index=False)

print("done!!!")
