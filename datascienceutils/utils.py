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
from statsmodels.api.tsa.statespace import SARIMAX
from statsmodels.api.tsa import AR
# Online classifiers http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
# xgboost
import xgboost as xgb
# LightGBM models
from pylightgbm.models import *
# Sigh lightgbm insist this is the only way
os.environ['LIGHTGBM_EXEC'] = os.path.join(os.getenv("HOME"), 'bin', 'lightgbm')

# (Gaussian) Mixture models
from sklearn.mixture import *

models_dict = { 'knn': KNeighborsClassifier,
                'gaussianNB': GaussianNB,
                'multinomialNB': MultinomialNB,
                'bernoulliNB': BernoulliNB,
                'randomForest': RandomForestClassifier,
                'tree': DecisionTreeClassifier,
                'svm': SVC,
                'LinearRegression': LinearRegression,
                'RidgeRegression': RidgeRegression,
                'RidgeRegressionCV': RidgeRegressionCV,
                'LassoRegression': LassoRegression,
                'ElasticNetRegression': ElasticNet,
                'LogisticRegression': LogisticRegression,
                'RANSACRegression': RANSACRegressor,
                'pca': PCA,
                'lda': LinearDiscriminantAnalysis,
                'tsne': TSNE,
                'kde': KernelDensity,
                'AR': AR,
                'SARIMAX': SARIMAX,
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


def get_model_obj(modelType, n_clusters=None, **kwargs):
    global models_dict
    if modelType in models_dict:
        return models_dict[modelType](**kwargs)
    else:
        raise 'Unknown model type: see utils.py for available'



