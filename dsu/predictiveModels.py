import numpy as np
import os

from collections import defaultdict
from sklearn import model_selection, metrics

# Custom utils
from . import utils
from . import settings

def cross_val_train(dataframe, target, modelType, **kwargs):
    cv = kwargs.pop('cv',None)
    model = utils.get_model_obj(modelType, **kwargs)
    scores = cross_val_score(model, dataframe, target, cv=cv)
    return scores

def train(dataframe, target, modelType, column=None, **kwargs):
    """
    Generic training wrapper around different scikits-learn models

    @params:
        @dataframe: A pandas dataframe with all feature columns.
        @target: pandas series or numpy array(basically a iterable) with the target values. should match length with dataframe
        @modelType: String representing the model you want to train with

    @return:
        Model object
    """
    model = utils.get_model_obj(modelType, **kwargs)
    if column:
        source = dataframe[column].values.reshape(-1, 1)
        model.fit(source, target)
    else:
        model.fit(dataframe, target)
    return model

def grid_search(dataframe, target, modelType, **kwargs):
    model = utils.get_model_obj(modelType, **kwargs)
    scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
    clf = model_selection.GridSearchCV(model, scoring=scorer, cv=2)
    clf.fit(dataframe, target)
    return clf

def dump_results(results_df, filename, model_params, kaggle=True):
    assert kaggle, 'only supporting kaggle format'
    final_fnam = utils.get_full_path(settings.RESULTS_BASE_PATH,
                                          filename, model_params, extn='.csv')
    results_df.to_csv(final_fnam, index=False)
    utils.call_7z(final_fnam)

def featureSelect(dataframe):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return sel.fit_transform(dataframe)

