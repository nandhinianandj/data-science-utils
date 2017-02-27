import logging
import math
import os

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift,\
                            Birch, AffinityPropagation, AgglomerativeClustering

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
# For svm models
from sklearn.svm import *
# For regression models
from sklearn.linear_model import *
# Dimensionality reduction/ factor analysis models
# PCA
from sklearn.decomposition import PCA
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# tsne
from sklearn.manifold import TSNE
# kernel density estimators
from sklearn.neighbors.kde import KernelDensity
# TIME SERIES MODELS
import statsmodels.api as sm
# Online classifiers http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
# xgboost
import xgboost as xgb
# LightGBM models
from pylightgbm.models import *
# Sigh lightgbm insist this is the only way
os.environ['LIGHTGBM_EXEC'] = os.path.join(os.getenv("HOME"), 'bin', 'lightgbm')

# (Gaussian) Mixture models
from sklearn.mixture import *

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
                'pca': PCA,
                'lda': LinearDiscriminantAnalysis,
                'tsne': TSNE,
                'kde': KernelDensity,
                'AR': sm.tsa.AR,
                'SARIMAX': sm.tsa.statespace.SARIMAX,
                'sgd': SGDClassifier,
                'perceptron': Perceptron,
                'xgboost': xgb.XGBClassifier,
                'baseNN': create_base_nn,
                'lightGBMRegression': GBMRegressor,
                'lightGBMBinaryClass': GBMClassifier,
                'KMeans':  KMeans,
                'dbscan': DBSCAN,
                'affinity_prop': AffinityPropagation,
                'spectral': SpectralClustering,
                'birch': Birch,
                'agglomerativeCluster': AgglomerativeClustering,
                'meanShift': MeanShift,
                'gmm': GaussianMixture,
                'bgmm': BayesianGaussianMixture,

        }


# Type checkers taken from here. http://stackoverflow.com/questions/25039626/find-numeric-columns-in-pandas-python
def is_type(df, baseType):
    import numpy as np
    import pandas as pd
    test = [issubclass(np.dtype(d).type, baseType) for d in df.dtypes]
    return pd.DataFrame(data = test, index = df.columns, columns = ["test"])

def calculate_anova(df, targetCol, sourceCol):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    lm = ols('%s ~ C(%s, Sum) + c'% (targetCol, sourceCol),
            data=df).fit()
    table = anova_lm(lm, typ=2)
    return table

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def is_float(df):
    import numpy as np
    return is_type(df, np.float)

def is_number(df):
    import numpy as np
    return is_type(df, np.number)

def is_integer(df):
    import numpy as np
    return is_type(df, np.integer)

def chunks(combos, size=9):
    for i in range(0, len(combos), size):
        yield combos[i:i + size]

def roundup(x):
    """
    :param x:
    :return: round up the value
    """
    return int(ceil(x / 10.0))*2

def get_model_obj(modelType, n_clusters=None, **kwargs):
    global models_dict
    if modelType in models_dict:
        return models_dict[modelType](**kwargs)
    else:
        raise 'Unknown model type: see utils.py for available'


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

