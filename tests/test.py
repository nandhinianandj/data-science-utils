# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : test.py
#
#* Purpose :
#
#* Creation Date : 28-04-2020
#
#* Last Modified : Tuesday 28 April 2020 02:26:53 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

import unittest



def test_linear_regression_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.regression_analyze(dataframe, numCols, check_vif=True, check_heteroskedasticity=True)

def test_non_linear_regression_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.non_linear_regression_analyze(dataframe, numCols, check_vif=True, check_heteroskedasticity=True)

def test_bayesian_regression_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.bayesian_regression_analyze(dataframe, numCols,)

def test_visual_analyze():
    from dsu import analyze
    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.regression_analyze(dataframe, numCols, check_vif=True, check_heteroskedasticity=True)

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
