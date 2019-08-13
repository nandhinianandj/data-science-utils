import sklearn.ensemble as ske
#from sklearn.feature_selection import VarianceThreshold
#import statsmodels.api as sm
#from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel

# Local import
from .nlp import *
from .vision import *
from .audio import *


# def vif_filter(df):
#
#
#
#__all__ = ['word_match_share', 'word_2_vector', 'word_cloud']
#TODO: https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
#TODO: filter of categorical features by how much of the dataset they divide the records into.
#TODO: filter of numerical features by variance of the features.
def pick_features(df, targetCol):
    y = df[targetCol]
    df.drop(targetCol, inplace=True)
    X = df
    fsel = ske.ExtraTreesClassifier().fit(X, y)
    model = SelectFromModel(fsel, prefit=True)
    return model.transform(X)

#
# def pickFeatures(df, targetCol, exclude_columns=[], categories=[]):
#     columns = set(filter(lambda x: x not in exclude_columns, df.columns))
#     responseType = df.targetCol.dtype.name
#     numericalColumns = set(df.select_dtypes(include=[np.number]).columns).intersection(columns)
#     if responseType != 'category':
#         anovaTable = dict()
#         for each in categories:
#             anovaTable[each] = calculate_anova(df[targetCol])
#     else:
#         pass
#
# def pca_univariate_pick(df, target):
#
#     # This dataset is way too high-dimensional. Better do PCA:
#     pca = PCA(n_components=2)
#
#     # Maybe some original features where good, too?
#     selection = SelectKBest(k=1)
#
#     # Build estimator from PCA and Univariate selection:
#
#     combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#
#     # Use combined features to transform dataset:
#     df_features = combined_features.fit(df, target).transform(df)
#
#     svm = SVC(kernel="linear")
#
#     # Do grid search over k, n_components and C:
#
#     pipeline = Pipeline([("features", combined_features), ("svm", svm)])
#
#     param_grid = dict(features__pca__n_components=[1, 2, 3],
#                               features__univ_select__k=[1, 2],
#                                                 svm__C=[0.1, 1, 10])
#
#     grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
#     return grid_search.fit(df, target)
