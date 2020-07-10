# Standard and external libraries
import json
import pprint
import statsmodels
from tqdm import tqdm_notebook

import operator
import itertools
import random
import numpy as np
import pandas as pd

# Custom libraries
from . import sklearnUtils as sku
from . import plotter
from . import utils
from . import statsutils as su


def missing_values(df, sample=False, samplesize=1000, groups=None):
    if groups:
        for each in groups:
            groupdf = df.groupby(each).apply(pd.isnull)
            plot_missing_vals(groupdf, sample=sample, samplesize=samplesize)
    else:
        plot_missing_vals(df, sample=sample, samplesize=samplesize)

def plot_missing_vals(df, sample=True, samplesize=1000):
    import missingno as msno
    if not sample:
        msno.bar(df)
        msno.heatmap(df)
        msno.dendrogram(df)
        msno.matrix(df)
    else:
        msno.bar(df.sample(samplesize))
        msno.heatmap(df.sample(samplesize))
        msno.dendrogram(df.sample(samplesize))
        msno.matrix(df.sample(samplesize))

def dist_analyze( df, column=None,
                  category=None, is_normal=True,
                  bayesian_hist=False, kdeplot=True,
                  violinplot=True):
    """
    Analyze/plot visual of a single distribution
    @params:
        @column: presence assumes this is one numeric column
        @category:
    """
    plots = list()
    if column:
        assert utils.is_numeric(df, column=column)
        print("Variance of %s" % column)
        print(df[column].var())
        print("Skewness of %s" % column)
        print(df[column].skew())
        if is_normal:
            su.check_normality(df[column], column)

        if violinplot:
            plots.append(plotter.sb_violinplot(df[column], inner="box"))
        plots.append(
            plotter.sb_histogram(
                df[column], kdeplot=kdeplot, bayesian_bins=bayesian_hist
            )
        )
        plotter.sb_show()

    if kdeplot:
        assert utils.is_numeric(
            df, column=column
        ), "KDE Plot Only available for numerical columns"
        df[column].plot.kde(label=column, title="Kernel density estimate")

    if category:
        plots.append(
            plotter.sb_barplot(df, category, column)
        )  # , title='Distribution of %s'%column))
        plotter.sb_show()
    return plots


def joint_dist_analyze(df, columns=None, categories=None):
    for category in categories:
        for column in columns:
            assert utils.is_numeric(df, column=column)
            print("Variance of %s" % column, df[column].var())
            print("Skewness of %s" % column, df[column].skew())
            # Plot Barplots of combination of category and numerical columns
            plotter.sb_barplot(df, category, column)
            plotter.sb_show()
    #grid = plotter.gridplot(list(utils.chunks(plots, size=2)))
    #return grid


def outliers_analyze(df, **kwargs):

    rng = np.random.RandomState(42)

    # Example settings
    n_samples = kwargs.get('samples', None) or 200
    outliers_fraction = kwargs.get('outliers_fraction', None) or 0.25
    clusters_separation = [0, 1, 2]

    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(
            nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1
        ),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(
            max_samples=n_samples, contamination=outliers_fraction, random_state=rng
        ),
    }
    plots = list()
    for i, (clf_name, clf) in tqdm_notebook(enumerate(classifiers.items())):
        # fit the data and tag outliers
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
        y_pred = clf.predict(X)
        predictions[clf_name] = y_pred
        n_errors = (y_pred != ground_truth).sum()
        # plot the levels lines and the points
        # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        # plots.append(plotter.contourplot(xx, yy, Z,
        #                                levels=np.linspace(Z.min(), threshold, 7)))
    # return plotter.grid_plot(plots)
    return predictions


def correlation_analyze( df, col1, col2,
                         categories=[], measures=[], summary_only=False,
                         check_linearity=False, trellis=False,):
    """
    Visually plot possible correlation between two columns mentioned and other categories and
    measures involved
    If categories and measures are passed, plot heatmap of combination of categories by measure.

    @params:
        df: Dataframe table data.
        categories: list of categorical variable names
        measures: List of measures to plot heatmap of categories
    """
    if summary_only:
        plotter.sb_heatmap(df.corr())
        return
    for meas in measures:
        assert meas in list(df.columns)
    for catg in categories:
        assert catg in list(df.columns)

    # TODO: add check for multi-collinearity
    if categories and not measures:
        measures = ["count"]

    u, v = col1, col2
    plotter.sb_scatterplot(df[u], df[v])
    plotter.sb_show()

    # plotter.sb_jointplot(df[u], df[v])
    if check_linearity:
        u_2diff = np.gradient(df[u], 2)
        v_2diff = np.gradient(df[v], 2)
        print("Linearity btw %s and %s" % (u, v))
        print("No. of 2nd differences: %d" % len(u_2diff))
        linearity_2nd_diff = np.divide(u_2diff, v_2diff)
        # Drop inf and na values
        linearity_2nd_diff = linearity_2nd_diff[~np.isnan(linearity_2nd_diff)]
        linearity_2nd_diff = linearity_2nd_diff[~np.isinf(linearity_2nd_diff)]
        print(np.mean(linearity_2nd_diff))

    # print("# Correlation btw Numerical Columns")
    # plotter.gridplot(list(utils.chunks(plots, size=2)))

    if categories and measures:
        # Plot heatmaps of category-combos by measure value.
        heatmaps = []
        catCombos = itertools.combinations(categories, 2)
        cols = list(df.columns)
        if "count" in measures:
            # Do a group by on categories and use count() to heatmap
            measures.remove("count")
            for combo in catCombos:
                print(
                    "# Correlation btw Columns %s & %s by count" % (combo[0], combo[1])
                )
                group0 = df.groupby(list(combo)).size().reset_index()
                group0.rename(columns={col1: "counts"}, inplace=True)
                plotter.sb_heatmap(group0, combo[0], combo[1], "counts")
                plotter.sb_show()

        for meas in measures:
            # Plot heatmaps for measure across all combination of categories
            for combo in catCombos:
                print(
                    "# Correlation btw Columns %s & %s by measure %s"
                    % (combo[0], combo[1], meas)
                )
                group0 = df.groupby(list(combo)).sum().reset_index()
                group0.rename(columns={meas: "sum_%s" % meas}, inplace=True)
                plotter.sb_heatmap(
                    group0,
                    combo[0],
                    combo[1],
                    "sum_%s" % meas,
                    title="%s vs %s %s heatmap" % (combo[0], combo[1], meas),
                )
                plotter.sb_show()


def non_linear_regression_analyze(df, target_cols=list(), trainsize=0.8, **kwargs):

    plots = list()
    for col1, col2 in itertools.combinations(target_cols, 2):
        from ace import model

        model = model.Model()
        model.build_model_from_xy([df[col1].values], [df[col2].values])

        print(" # Ace Models btw numerical cols")
        plot = plotter.sb_lineplot(col1, col2, df[[col1, col2]])  # , title='%s Vs %s'%(col1, col2))
        plots.append(plot)
    plotter.show(plots)


def pymc_plot_posterior_glm(df, xlabel, ylabel, trace):
    from pymc3.plots import plot_posterior_predictive_glm
    import matplotlib.pyplot as plt

    x = df[xlabel].values
    y = df[ylabel].values
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, "x", label="data")

    plot_posterior_predictive_glm(
        trace, samples=100, label="posterior predictive regression lines"
    )
    plt.title("Posterior predictive regression lines")
    plt.legend(loc=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def pymc_regression_analyze(df, target_cols=list(), **kwargs):
    from pymc3 import traceplot
    import matplotlib.pyplot as plt

    for col1, col2 in itertools.combinations(target_cols, 2):
        new_df = df[[col1, col2]].copy(deep=True)
        target = new_df[col2]
        trace = utils.train_pymc_linear_reg(new_df, col2, column=col1)
        plt.figure(figsize=(7, 7))
        plt.title("%s Vs %s" % (col1, col2))
        traceplot(trace[100:])
        plt.tight_layout()
        pymc_plot_posterior_glm(df, col1, col2, trace)


def pymc_log_reg_analyze(dataframe, target_col=str(), **kwargs):
    import pymc3 as pm

    other_cols = set(dataframe.columns) - target_col
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula(
            "income ~ age + age2 + educ + hours",
            data,
            family=pm.glm.families.Binomial(),
        )
        trace = pm.sample(1000, tune=1000, init="adapt_diag")


def regression_analyze( df, target_cols=list(),
                        trainsize=0.8, check_heteroskedasticity=True,
                        check_vif=True, **kwargs):
    """
    Plot regressed data vs original data for the passed columns.
    @params:
        col1: x column,
        col2: y column

    @optional:
        non_linear: Use the python ace module to calculate non-linear correlations too.(Warning can
        be very slow)
        check_heteroskedasticity: self-evident
    """
    from . import predictiveModels as pm

    # this is the quantitative/hard version of teh above
    # TODO: Simple  line plots of column variables, but for the y-axis,
    # Fit on
    #         b, logarithmic/exponetial function
    #         c, logistic function
    #         d, parabolic function??
    #   Additionally plot the fitted y and the correct y in different colours against the same x

    from statsmodels.stats import outliers_influence as oi

    plots = list()
    results = list()
    finalRes = dict()
    for col1, col2 in itertools.combinations(target_cols, 2):
        new_df = df[[col1, col2]].copy(deep=True)
        target = new_df[col2].values.reshape(-1, 1)
        key = (col1, col2)
        finalRes[key] = dict()
        models = [
            pm.train(new_df, target, column=None, modelType="LinearRegression"),
            pm.train(new_df, target, column=None, modelType="RidgeRegression"),
            pm.train(new_df, target, column=None, modelType="RidgeRegressionCV"),
            pm.train(new_df, target, column=None, modelType="LassoRegression"),
            pm.train(new_df, target, column=None, modelType='ElasticNetRegression'),
            pm.train(new_df, target, column=None, modelType='SVMRegression'),
            pm.train(new_df, target, column=col1, modelType='IsotonicRegression'),
            #pm.train(new_df, target, column=col1, modelType='logarithmicRegression'),
        ]
        tempRes = dict()
        if check_heteroskedasticity:
            tempRes["HeteroSkedas_%s" % col1] = list()
            resHetSkedas = statsmodels.stats.diagnostic.het_breuschpagan(
                target, exog_het=df[[col1, col2]]
            )
            print("Hetero-Skedasticity test(Breush-Pagan)")
            print(resHetSkedas)
            tempHetSkedas = {
                "HeteroSkedasticity": dict(
                    zip(["lm", "lm_pvalue", "fvalue", "f_pvalue"], resHetSkedas)
                )
            }
            tempRes["HeteroSkedas_%s" % col1].append(tempHetSkedas)
        for model in models:
            modelName = model.__repr__().split("(")[0]
            tempRes[modelName] = list()
            scatter = plotter.sb_scatterplot( new_df[col1], new_df[col2])  # , plttitle=model.__repr__())
            predicted = list(model.predict(new_df))
            print("Regression Score: %s" % (model.__repr__()))
            regScore = model.score(new_df, new_df[col2].values.reshape(-1, 1))
            print(regScore)
            tempRes[modelName].append({"score": regScore})
            # tempRes[modelName].append({'rank':model.rank_})
            tempRes[modelName].append({"intercept": model.intercept_})
            tempRes[modelName].append({"coef": model.coef_})
            if check_vif:
                tempVIF = list()
                tempRes[modelName].append({"VIF": []})

                for col in [col1, col2]:
                    col_idx = list(df.columns).index(col)
                    vifFactor = oi.variance_inflation_factor(
                        df.values.reshape(-1, 1), col_idx
                    )
                    tempVIF.append({col: vifFactor})
                    print("Variance Inflation Factors for %s" % col)
                    print(vifFactor)
                    tempRes[modelName]["VIF"].append({col: vifFactor})

        finalRes[key] = tempRes
    pprint.pprint(finalRes)
    return finalRes
    # grid = plotter.gridplot(list(utils.chunks(plots, size=2)))
    # plotter.show(grid)

def time_series_analysis(   df, timeCol="date", valueCol=None,
                            timeInterval="30min", plot_title="timeseries",
                            skip_stationarity=False, skip_autocorrelation=False,
                            skip_seasonal_decompose=False, psf_analyze=False, **kwargs):
    """
    Plot time series, rolling mean, rolling std , autocorrelation plot, partial autocorrelation plot
    and seasonal decompose
    """
    # TODO: implement R-PSF forecasting interface
    # TODO: integrate features from fbprophet
    from . import timeSeriesUtils as tsu

    if "create" in kwargs:
        ts = tsu.create_timeseries_df(
            df, timeCol=timeCol, timeInterval=timeInterval, **kwargs.get("create")
        )
    else:
        ts = tsu.create_timeseries_df(df, timeCol=timeCol, timeInterval=timeInterval)
    # TODO;
    # 3. ARIMA model of the times
    # 4. And other time-series models like AR, etc..
    # 5. Wrappers around fbprophet
    if "stationarity" in kwargs:
        plot = tsu.test_stationarity(
            ts,
            timeCol=timeCol,
            valueCol=valueCol,
            title=plot_title,
            skip_stationarity=skip_stationarity,
            **kwargs.get("stationarity")
        )
    else:
        plot = tsu.test_stationarity(
            ts,
            timeCol=timeCol,
            valueCol=valueCol,
            title=plot_title,
            skip_stationarity=skip_stationarity,
        )
        plotter.show(plot)
    if not skip_autocorrelation:
        if "autocorrelation" in kwargs:
            tsu.plot_autocorrelation(
                ts, valueCol=valueCol, **kwargs.get("autocorrelation")
            )  # AR model
            tsu.plot_autocorrelation(
                ts, valueCol=valueCol, partial=True, **kwargs.get("autocorrelation")
            )  # partial AR model
        else:
            tsu.plot_autocorrelation(ts, valueCol=valueCol)  # AR model
            tsu.plot_autocorrelation(
                ts, valueCol=valueCol, partial=True
            )  # partial AR model

    if not skip_seasonal_decompose:
        if "seasonal" in kwargs:
            seasonal_args = kwargs.get("seasonal")
            tsu.seasonal_decompose(ts, **seasonal_args)
        else:
            tsu.seasonal_decompose(ts)

def causal_analyze(dataframe):
    for each in dataframe.columns:
        assert utils.is_numeric(dataframe, column=each)
    import networkx
    from networkx import drawing
    import causality
    from causality.inference.independence_tests import RobustRegressionTest
    from causality.inference.search import IC

    variable_types = dict(zip(dataframe.columns, ["c"] * len(dataframe)))
    ic_algorithm = IC(RobustRegressionTest)
    graph = ic_algorithm.search(dataframe, variable_types)
    drawing.draw_networkx(graph)
    pass

