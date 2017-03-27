# Standard and external libraries
from bokeh.layouts import gridplot
from statsmodels.stats import diagnostic

import operator
import functools
import itertools
import random
import numpy as np
import pandas as pd

# Custom libraries
from . import sklearnUtils
from . import plotter
from . import utils

#TODO: only the non-parametric ones used, check the rest andfigure out how to choose
# parameters(think kde estimator from sklearn)
CHECK_DISTS = ['norm']#, 'zipf', 'geom', 'hypergeom', 'poisson', 'randint',
        #'multivariate_normal', 'weibull_min', 'weibull_max', 'logistic', 'chi', 'chi2', 'cosine',
        #'cauchy','alpha', 'beta', ] #'bernoulli','binom',

def distribution_tests(df, column, test_type='ks'):
    from scipy import stats
    for distribution in CHECK_DISTS:
        if test_type=='ks':
            print("Kolmogrov - Smirnov test with distribution %s"%distribution)
            print(stats.kstest(df[column].tolist(), distribution))
        #elif test_type =='wald':
        #    print("Wald test with distribution %s"%distribution)
        #    print(lm.wald_test(df[column], distribution))
        else:
            raise "Unknow distribution similarity test type"

def check_normality(series, name):
    print("Anderson-Darling normality test on %s "%name)
    print("Statistic: %d \n p-value: %d\n"%diagnostic.normal_ad(series))

def dist_analyze(df, column='', category='', is_normal=True, bayesian_hist=False):
    plots = []
    if (utils.is_numeric(df, column=column)):
        print("Variance of %s"%column)
        print(df[column].var())
        print("Skewness of %s"%column)
        print(df[column].skew())
        distribution_tests(df, column)
        if is_normal:
            check_normality(df[column], column)
        plots.append(plotter.sb_violinplot(df[column], inner='box'))
        if bayesian_hist:
            plots.append(plotter.histogram(df, column, bayesian_bins=True))
    else:
        if df[column].nunique() < 7:
            plots.append(plotter.pieChart(df, column, title='Distribution of %s'%column))
        else:
            print("Too many categories for col: %s can't plot pie-chart"%column)

    if category:
        # Plot Barplots of combination of category and numerical columns
        plots.append(plotter.barplot(df, column, category))
        print("# Joint Distribution of Numerical vs Categorical Columns")
    grid = gridplot(list(utils.chunks(plots, size=2)))
    return grid


def correlation_analyze(df, col1, col2, categories=[], measures=[],
                        check_linearity=False, trellis=False):
    """
    Plot scatter plots of all combinations of numerical columns.
    If categories and measures are passed, plot heatmap of combination of categories by measure.

    @params:
        df: Dataframe table data.
        categories: list of categorical variable names
        measures: List of measures to plot heatmap of categories
        trellis: Plot trellis type plots for the categories only valid if categories is passed
    """
    if summary_only:
        plotter.heatmap(df.corr())
    for meas in measures:
        assert meas in list(df.columns)
    for catg in categories:
        assert catg in list(df.columns)

    #TODO: add check for multi-collinearity
    if categories and not measures:
        measures = ['count']

    # Plot scatter plot of combination of numerical columns
    plots = []

    u,v = col1, col2
    plots.append(plotter.scatterplot(df, u, v))
    plots.append(plotter.sb_jointplot(df[u], df[v]))
    if check_linearity:
        u_2diff = np.gradient(df[u], 2)
        v_2diff = np.gradient(df[v], 2)
        print("Linearity btw %s and %s"%(u, v))
        print("No. of 2nd differences: %d"%len(u_2diff))
        linearity_2nd_diff = np.divide(u_2diff, v_2diff)
        # Drop inf and na values
        linearity_2nd_diff = linearity_2nd_diff[~np.isnan(linearity_2nd_diff)]
        linearity_2nd_diff = linearity_2nd_diff[~np.isinf(linearity_2nd_diff)]
        print(np.mean(linearity_2nd_diff))

    print("# Correlation btw Numerical Columns")
    grid = gridplot(list(utils.chunks(plots, size=2)))
    plotter.show(grid)

    if (categories and measures):
        # Plot heatmaps of category-combos by measure value.
        heatmaps = []
        combos = itertools.combinations(categories, 2)
        cols = list(df.columns)
        if 'count' in measures:
            # Do a group by on categories and use count() to heatmap
            measures.remove('count')
            for combo in combos:
                print("# Correlation btw Columns %s & %s by count" % (combo[0], combo[1]))
                group0 = df.groupby(list(combo)).size().reset_index()
                group0.rename(columns={0: 'counts'}, inplace=True)
                heatmaps.append(plotter.heatmap(group0, combo[0], combo[1], 'counts'))

        for meas in measures:
            # Plot heatmaps for measure across all combination of categories
            for combo in combos:
                print("# Correlation btw Columns %s & %s by measure %s" % (combo[0],
                    combo[1],
                    meas))
                group0 = df.groupby(list(combo)).sum().reset_index()
                group0.rename(columns={0: 'sum_%s'%meas}, inplace=True)
                heatmaps.append(plotter.heatmap(group0, combo[0], combo[1], 'sum_%s'%meas,
                                                title="%s vs %s %s heatmap"%(combo[0], combo[1], meas)
                                                ))
        hmGrid = gridplot(list(utils.chunks(heatmaps, size=2)))
        plotter.show(hmGrid)
        if trellis:
            trellisPlots = list()
    #print("# Pandas correlation coefficients matrix")
    #print(df.corr())
    #print("# Pandas co-variance coefficients matrix")
    #print(df.cov())

def is_independent(series1, series2):
    pass

def is_similar_distribution(origin_dist, target_dist):
    import permutation_test as p
    p_value = p.permutation_test(data, ref_data)
    print(p_value)

def degrees_freedom(df, dof_range = [], categoricalCol=[]):
    """
    Find what are the maximum orthogonal dimensions in the data
    """
    if categoricalCol:
        assert len(categoricalCol)==2, "Only two categories supported"
        probabilities = dict()
        for col in categoricalCol:
            values = df[categoricalCol].unique()
            grouped_df = df.groupby(categoricalCol).count()
            for val in values:
                probabilities[(col,val)] = grouped_df[val]/df[categoricalCol].count()
        print(probabilities)
    else:
        print("Chi-square test of independence()")
        from scipy.stats import chi2_contingency
        result = chi2_contingency(df.as_matrix())
        print("Statistical degrees of freedom")
        print(result[2])
        print("Chi-square value")
        print(result[0])
        print("p-value")
        print(result[1])

def cosine_distance():
    assert hasattr(dof_range, '__iter__')
    # TODO: Extend/generalise this to more than 2-norm (aka 2-D plane)
    from scipy import spatial
    dof_range = [2]
    all_cosine_dists = dict()
    for each in dof_range:
        combos = itertools.combinations(df.columns, each)
        # TODO: calculate cosine distance
        cosine_dist = dict()
        for combo in combos:
            cosine_dist[combo] = spatial.distance.cosine(df[combo[0]], df[combo[1]])
        all_cosine_dists[each] = sorted(cosine_dist.items(), key=operator.itemgetter(1))
    print("Cosine Distance Method")
    return all_cosine_dists

def factor_analyze(df, target=None, model_type ='pca', **kwargs):
    model = utils.get_model_obj(model_type, **kwargs)
    numericalColumns = df.select_dtypes(include=[np.number]).columns
    catColumns = set(df.columns).difference(set(numericalColumns))
    for col in catColumns:
        df[col] = sklearnUtils.encode_labels(df, col)
    print("Model being used is :%s "%model_type)
    if model_type == 'linear_da':
        assert target is not None, "Target class/category necessary for Linear DA factor analysis"
        model.fit(df, target)
        print("Coefficients")
        print(model.coef_)
        print("Covariance")
        print(model.covariance_)
    elif model_type == 'latent_da':
        print("Components")
        print(model.components_)
    else:
        model.fit(df[numericalColumns])
        print("No. of Components")
        print(model.n_components)
        print("Components")
        print(model.components_)
        print("Explained variance")
        print(model.explained_variance_)
    trans_df = pd.DataFrame(model.transform(df))

    print("Correlation of transformed")
    correlation_analyze(trans_df, 0, 1)

def regression_analyze(df, col1, col2, trainsize=0.8, non_linear=False,
                       test_heteroskedasticity=False, check_dist_similarity=False, **kwargs):
    """
    Plot regressed data vs original data for the passed columns.
    @params:
        col1: x column,
        col2: y column

    @optional:
        non_linear: Use the python ace module to calculate non-linear correlations too.(Warning can
        be very slow)
        test_heteroskedasticity: self-evident
    """
    # TODO: non-linearity tests
    from . import predictiveModels as pm


    # this is the quantitative/hard version of teh above
    #TODO: Simple  line plots of column variables, but for the y-axis,
    # Fit on
    #         b, logarithmic/exponetial function
    #         c, logistic function
    #         d, parabolic function??
    #   Additionally plot the fitted y and the correct y in different colours against the same x

    if non_linear:
        plots = list()
        import ace
        model = ace.model.Model()
        model.build_model_from_xy([df[col1].as_matrix()], [df[col2].as_matrix()])

        print(" # Ace Models btw numerical cols")
        plot = plotter.lineplot(df[[col1, col2]], col1, col2)
        plotter.show(plot)

    if check_dist_similarity:
        is_similar_distribution(df[col1], df[col2])

    new_df = df[[col1, col2]].copy(deep=True)
    target = new_df[col2]
    models = [
            pm.train(new_df, target, column=col1, modelType='LinearRegression'),
            pm.train(new_df, target, column=col1, modelType='RidgeRegression'),
            pm.train(new_df, target, column=col1, modelType='RidgeRegressionCV'),
            pm.train(new_df, target, column=col1, modelType='LassoRegression'),
            pm.train(new_df, target, column=col1, modelType='ElasticNetRegression'),
            #pm.train(new_df, target, column=col1, modelType='IsotonicRegression'),
            #pm.train(new_df, target, column=col1, modelType='logarithmicRegression'),
            ]
    plots = list()
    for model in models:
        scatter = plotter.scatterplot(new_df, col1, col2, plttitle=model.__repr__())
        source = new_df[col1].as_matrix().reshape(-1,1)
        predicted = list(model.predict(source))
        flatSrc = [item for sublist in source for item in sublist]
        scatter.line(flatSrc, predicted,
                     line_color='red')
        plots.append(scatter)
        print("Regression Score: %s"%(model.__repr__()))
        print(model.score(source, new_df[col2].as_matrix().reshape(-1,1)))
        if test_heteroskedasticity:
            if not kwargs.get('exog', None):
                other_cols = list(set(df.columns) - set([col1, col2]))
                kwargs['exog'] = random.choice(other_cols)
            exog = df[kwargs.get('exog')].as_matrix().reshape(-1,1)
            print("Hetero-Skedasticity test(Breush-Pagan)")
            print(diagnostic.het_breushpagan(model.residues_, exog_het=exog))
    grid = gridplot(list(utils.chunks(plots, size=2)))
    plotter.show(grid)

def time_series_analysis(df, timeCol='date', valueCol=None, timeInterval='30min',
        plot_title = 'timeseries',
        skip_stationarity=False,
        skip_autocorrelation=False,
        skip_seasonal_decompose=False, **kwargs):
    """
    Plot time series, rolling mean, rolling std , autocorrelation plot, partial autocorrelation plot
    and seasonal decompose
    """
    from . import timeSeriesUtils as tsu
    if 'create' in kwargs:
        ts = tsu.create_timeseries_df(df, timeCol=timeCol, timeInterval=timeInterval, **kwargs.get('create'))
    else:
        ts = tsu.create_timeseries_df(df, timeCol=timeCol, timeInterval=timeInterval)
    # TODO;
    # 2. Seasonal decomposition of the time series and plot it
    # 3. ARIMA model of the times
    # 4. And other time-serie models like AR, etc..
    # 5. Wrappers around fbprophet
    if 'stationarity' in kwargs:
        plot = tsu.test_stationarity(ts, timeCol=timeCol, valueCol=valueCol,
                title=plot_title,
                skip_stationarity=skip_stationarity,
                **kwargs.get('stationarity'))
    else:
        plot = tsu.test_stationarity(ts, timeCol=timeCol, valueCol=valueCol,
                title=plot_title,
                skip_stationarity=skip_stationarity
                )
        plotter.show(plot)
    if not skip_autocorrelation:
        if 'autocorrelation' in kwargs:
            tsu.plot_autocorrelation(ts, valueCol=valueCol, **kwargs.get('autocorrelation')) # AR model
            tsu.plot_autocorrelation(ts, valueCol=valueCol, partial=True, **kwargs.get('autocorrelation')) # partial AR model
        else:
            tsu.plot_autocorrelation(ts, valueCol=valueCol) # AR model
            tsu.plot_autocorrelation(ts, valueCol=valueCol, partial=True) # partial AR model

    if not skip_seasonal_decompose:
        if 'seasonal' in kwargs:
            seasonal_args = kwargs.get('seasonal')
            tsu.seasonal_decompose(ts, **seasonal_args)
        else:
            tsu.seasonal_decompose(ts)

def chaid_tree(dataframe, targetCol):
    import CHAID as ch
    columns = dataframe.columns
    columns = list(filter(lambda x: x not in [targetCol], dataframe.columns))
    print(ch.Tree.from_pandas_df(dataframe, columns, targetCol))
