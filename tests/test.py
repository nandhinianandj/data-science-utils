# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

# * File Name : test.py
#
# * Purpose :
#
# * Creation Date : 28-04-2020
#
# * Last Modified : Wednesday 29 April 2020 09:37:45 PM IST
#
# * Created By :

# _._._._._._._._._._._._._._._._._._._._._.#

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import pandas as pd
import numpy as np
try:
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
except (ImportError, Exception):
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

dataframe = pd.DataFrame(X)
dataframe["target"] = y


def test_linear_regression_analyze():
    from dsu import analyze

    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.regression_analyze(
        dataframe, numColumns, check_vif=False, check_heteroskedasticity=True
    )


def test_non_linear_regression_analyze():
    from dsu import analyze

    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.non_linear_regression_analyze(
        dataframe, numColumns, check_vif=True, check_heteroskedasticity=True
    )


def test_bayesian_regression_analyze():
    from dsu import analyze

    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    analyze.pymc_regression_analyze(
        dataframe, numColumns,
    )


# def test_visual_analyze():
#    from dsu import analyze
#    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
#    analyze.regression_analyze(dataframe, numColumns, check_vif=True, check_heteroskedasticity=True)


def test_dimension_analyze():
    from dsu import analyze

    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    catCols = list(set(irisDf.columns) - set(numColumns))
    for col in catCols:
        analyze.dimension_analyze(dataframe, col)


def test_dist_analyze():
    from dsu import analyze

    numColumns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    for col in numColumns:
        analyze.dist_analyze(dataframe, col)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    print("Running test_linear_regression_analyze...")
    test_linear_regression_analyze()
    print("✓ test_linear_regression_analyze passed!")
    print("Running test_dist_analyze...")
    test_dist_analyze()
    print("✓ test_dist_analyze passed!")
