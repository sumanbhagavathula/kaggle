import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import distutils.dir_util
from scipy import stats

def plot_confusion_matrix(y_true,y_pred,labels,pathtosave = ''):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(labels)
    pred_labels = np.unique(labels)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    if pathtosave != '':
        plotfile = pathtosave + '/confusionmatrix.jpg'
        plt.savefig(plotfile)

    return

#helper function for plotting
def convergence_plot(f,data,labels, beta_vals, pathtosave = ''):
    n = beta_vals.shape[0]
    fs=np.zeros(beta_vals.shape[0])

    for i in range(0,n):
        #display(beta_vals[i,])
        fs[i] = f(beta_vals[i,],data,labels,0.1)
        print(fs[i])

    # Generate a sequence numbers from -10 to 10 with 100 steps in between
    x = np.linspace(1, n, n)

    # The plot function plots the first argument on the x-axis, the second argument on the y-axis and
    # connects the points.
    plt.plot(x, fs, marker="x")
    plt.xlabel('iteration')  # Add a label to the x-axis
    plt.ylabel('objective value')  # Add a label to the y-axis
    plt.title('Plot of Objective value with iteration')  # Adds a plot title

    if pathtosave != '':
        plotfile = pathtosave + '/convergenceplot.jpg'
        plt.savefig(plotfile)


#helper function for plotting
def plotcrossvalidationresults(hyperparams, scores, logbase = 10, pathtosave = ''):
    fig, ax = plt.subplots()
    ax.plot(np.log10(hyperparams), scores)
    plt.xlabel('log lambda')
    plt.ylabel('cross validation score')
    plt.title('mean cross validation score after folds vs. lambda=')

    if pathtosave != '':
        plotfile = pathtosave + '/crossvalidationresults.jpg'
        plt.savefig(plotfile)


def makeresultsdir(dirname):
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path='./results/' + dirname + '_' + dt
    distutils.dir_util.mkpath(path)

    return path

def isconverged(score, epsilon=.000001, max_iter=1000):
    isconverged = False

    if len(score) >= 30:
        mean=np.mean(score)
        score=score-mean
        #k,p=stats.mstats.normaltest(score)

        #if p>=0.05:
            #print('converged after ' + str(len(score)) + ' iterations')
        #    isconverged = True

        if np.std(score) <= epsilon:
            isconverged=True

    if len(score) == max_iter:
        print('forced convergence after ' + str(max_iter) + ' iterations')
        isconverged = True


    return isconverged