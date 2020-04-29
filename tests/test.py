# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : test.py
#
#* Purpose :
#
#* Creation Date : 28-04-2020
#
#* Last Modified : Wednesday 29 April 2020 01:51:56 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
X, y =  load_boston(return_X_y=True)
dataframe = pd.DataFrame(X)
dataframe['target'] = y

def test_linear_regression_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.regression_analyze( dataframe, numColumns,
                                check_vif=False, check_heteroskedasticity=True)

def test_non_linear_regression_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.non_linear_regression_analyze(dataframe, numColumns, check_vif=True, check_heteroskedasticity=True)

def test_bayesian_regression_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.bayesian_regression_analyze(dataframe, numColumns,)

def test_visual_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.regression_analyze(dataframe, numColumns, check_vif=True, check_heteroskedasticity=True)

def test_dimension_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    catCols = list(set(irisDf.columns) -set(numColumns))
    for col in catCols:
        analyze.dimension_analyze(dataframe, col)

def test_dist_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    for col in numColumns:
        analyze.dist_analyze(dataframe, col)

if __name__ == '__main__':
    test_linear_regression_analyze()
    test_non_linear_regression_analyze()
