import logging
import math
import os

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift,\
                            Birch, AffinityPropagation, AgglomerativeClustering, MiniBatchKMeans

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
# For svm models
from sklearn.svm import SVC, SVR
# For regression models
from sklearn.linear_model import *
from sklearn.isotonic import IsotonicRegression
# Dimensionality reduction/ factor analysis models
from sklearn.decomposition import PCA, NMF, FastICA, MiniBatchSparsePCA,\
                                    MiniBatchDictionaryLearning, FactorAnalysis
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# tsne
from sklearn.manifold import TSNE

# kernel density estimators
from sklearn.neighbors.kde import KernelDensity

# (Gaussian) Mixture models
from sklearn.mixture import *

# TIME SERIES MODELS
import statsmodels.api as sm
# Online classifiers http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
# xgboost
import xgboost as xgb

# Dimensionality Reduction with umap
# from umap import UMAP

# LightGBM models
#from pylightgbm.models import *
# Sigh lightgbm insist this is the only way
#os.environ['LIGHTGBM_EXEC'] = os.path.join(os.getenv("HOME"), 'bin', 'lightgbm')
os.environ['DATAROBOT_CONFIG_FILE'] = os.path.join(os.getenv("HOME"), '.config', 'datarobot',
        'drconfig.yaml')

# Scipy distance measures
#import neat

#def dummy_imports(name):
#    try:
#        eval('import ' + name)
#    except Exception:
#        global eval(name)

def create_base_nn(**kwargs):
    from keras.models import Sequential
    from keras.layers import Dense
    # create model
    model = Sequential()
    assert kwargs.get('inputParams', None)
    assert kwargs.get('outputParams', None)
    model.add(Dense(inputParams))
    model.add(Dense(outputParams))
    if kwargs.get('compileParams'):
        # Compile model
        model.compile(compileParams)# loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

models_dict = { 'knn': KNeighborsClassifier,
                'gaussianNB': GaussianNB,
                'multinomialNB': MultinomialNB,
                'bernoulliNB': BernoulliNB,
                'randomForest': RandomForestClassifier,
                'tree': tree.DecisionTreeClassifier,
                'svm': SVC,
                'LinearRegression': LinearRegression,
                'RidgeRegression': Ridge,
                'RidgeRegressionCV': RidgeCV,
                'LassoRegression': Lasso,
                'ElasticNetRegression': ElasticNet,
                'LogisticRegression': LogisticRegression,
                'RANSACRegression': RANSACRegressor,
                'IsotonicRegression': IsotonicRegression,
                'SVMRegression': SVR,

                'pca': PCA,
                'nmf': NMF,
                'FastICA': FastICA,
                'MiniBatchSparsePCA': MiniBatchSparsePCA,
                'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
                'MiniBatchKMeans': MiniBatchKMeans,
                'FactorAnalysis': FactorAnalysis,
                'lda': LinearDiscriminantAnalysis,
                'tsne': TSNE,
                #'umap': UMAP,

                'kde': KernelDensity,
                'AR': sm.tsa.AR,
                'SARIMAX': sm.tsa.statespace.SARIMAX,
                'sgd': SGDClassifier,
                'perceptron': Perceptron,
                'xgboost': xgb.XGBClassifier,
                'baseNN': create_base_nn,
                #'lightGBMRegression': GBMRegressor,
                #'lightGBMBinaryClass': GBMClassifier,
                'KMeans':  KMeans,
                'dbscan': DBSCAN,
                'affinity_prop': AffinityPropagation,
                'spectral': SpectralClustering,
                'birch': Birch,
                'agglomerativeCluster': AgglomerativeClustering,
                'meanShift': MeanShift,
                'gmm': GaussianMixture,
                'bgmm': BayesianGaussianMixture,
                #'pymc':
        }

def timestamp(datetimeObj):
    timestamp = (datetimeObj - datetime(1970, 1, 1)).total_seconds()
    return timestamp

def na_pct(series):
    assert isinstance(series, pd.Series)
    return 1 - series.count()/len(series)

# Type checkers taken from here. http://stackoverflow.com/questions/25039626/find-numeric-columns-in-pandas-python
def is_type(df, baseType, column=None, **kwargs):
    import numpy as np
    if not column:
        test = [issubclass(np.dtype(d).type, baseType) for d in df.dtypes]
        return pd.DataFrame(data = test, index = df.columns, columns = ["test"])
    else:
        return issubclass(np.dtype(df[column]).type, baseType)

def is_float(df, **kwargs):
    import numpy as np
    return is_type(df, np.float, **kwargs)

def is_number(df, **kwargs):
    import numpy as np
    return is_type(df, np.number, **kwargs)

def is_integer(df, **kwargs):
    import numpy as np
    return is_type(df, np.integer, **kwargs)

def is_numeric(series, **kwargs):
    if (is_number(series, **kwargs) or is_integer(series, **kwargs) or is_float(series,**kwargs)):
        return True
    return False

def chunks(combos, size=9):
    for i in range(0, len(combos), size):
        yield combos[i:i + size]

def roundup(x):
    """
    :param x:
    :return: round up the value
    """
    return int(ceil(x / 10.0))*2

def get_model_obj(modelType, **kwargs):
    global models_dict
    if modelType in models_dict:
        return models_dict[modelType](**kwargs)
    else:
        raise 'Unknown model type: see utils.py for available'

def train_pymc_linear_reg(df, target, column):
    from pymc3 import Model, glm, sample, NUTS, find_MAP
    mod = None
    with Model() as model:
        # specify glm and pass in data. The resulting linear model, its likelihood and
        # and all its parameters are automatically added to our model.
        data = dict(y=df[target].values, x=df[column].values)
        glm.GLM.from_formula('y~x', data)
        start = find_MAP()
        step = NUTS(scaling=start) # Instantiate MCMC sampling algorithm
        trace = sample(2000, step, progressbar=False) # draw 2000 posterior samples using NUTS sampling
    return trace

def cross_validate():
    for i, (train, test) in enumerate(cv):
        score = classifier.fit(dataframe[train], target[train]).decision_function(dataframe[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(target[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

def roc_plot(dataframe, target, score, cls_list=[],multi_class=True):
    import numpy as np
    import pandas as pd
    #import matplotlib.pyplot as plt
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from scipy import interp
    assert isinstance(target, (np.ndarray, pd.Series))
    # Not sure what this means some sort of initialization but are these right numbers?
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    num_classes = target.shape[1] or 1
    target = label_binarize(target, classes=cls_list)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    if not multi_class:
        #assert target.shape[1] == 1, "Please pass a nx1 array"
        #assert target.nunique() == 1, "Please pass a nx1 array"
        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        return plt
    else:
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 linewidth=2)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
                 linewidth=2)

        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
        return plt

def bins(df):
    """
    Thanks to guy from [here](https://stackoverflow.com/questions/16947336/binning-a-dataframe-in-pandas-in-python)
    """
    # Bin the data frame by index with 10 bins...
    assert is_numeric(df.index)
    bins = np.linspace(df.a.min(), df.a.max(), 10)
    groups = df.groupby(pandas.cut(df.a, bins))
    return groups

def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]

def get_full_path(base_path, filename, model_params, extn='.pkl', params_file=False):
    if params_file:
        return os.path.join(base_path, model_params['id'] + '_' + filename + '_' + 'params'+ extn)
    else:
        return os.path.join(base_path, model_params['id'] + '_' + filename + extn)

def call_7z(filename):
    import subprocess
    fname = os.path.basename(filename).split('.')[0]
    base_path = os.path.dirname(filename)
    cmd = ['7z', 'a', os.path.join(base_path, fname + '.7z'), os.path.join(base_path, filename), '-mx9']
    sp = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)


def count_boxes( data, box_size, M ):
    #data = Series( data )
    N = np.int( np.floor( M / box_size ) )
    counts = list()
    for i in range( N ):
        condition = ( data >= i*box_size )&( data < (i+1)*box_size )
        subset = data[ condition ]
        counts.append( subset.count() )
    counts = [ i for i in counts if i != 0 ]
    return len( counts )

def fractaldim(pointlist, boxlevel):
    """Returns the approximate fractal dimension of pointlist,
    via the box-counting algorithm.  The elements of pointset
    should be three-element sequences of numbers between 0.0
    and 1.0.  The boxlevel is the number of divisions made on
    each dimension, and should be greater than 1."""
    # From here https://mail.python.org/pipermail/python-list/2000-September/025263.html

    if boxlevel <= 1.0: return -1

    pointdict = {}

    def mapfunction(val, level=boxlevel):
            return int(val * level)

    for point in pointlist:
            box = (int(point[0] * boxlevel),
                    int(point[1] * boxlevel),
                    int(point[2] * boxlevel))
            #box = tuple(map(mapfunction, point))
            if not pointdict.has_key(box):
                    pointdict[box] = 1

    num = len(pointdict.keys())

    if num == 0: return -1

    return math.log(num) / math.log(boxlevel)

def memoize(func):
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func
