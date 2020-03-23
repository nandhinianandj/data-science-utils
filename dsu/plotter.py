# Standard and External lib imports
from pandas.api import types as ptypes
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import ( Text, PanTool, WheelZoomTool, LinearAxis,
                           SingleIntervalTicker, Range1d,  Plot,
                           Text, Circle, HoverTool, Triangle)
from math import ceil

from IPython.display import display

import matplotlib
from matplotlib import pyplot as plt
import itertools
import numpy as np
import operator
import os
import pandas as pd
import random
import seaborn as sns

sns.set(style='white')

#TODO: Ugh.. this file/module needs a cleanup
# Custom imports
from . import utils

def genColors(n, ptype=None):
    """
    """
    from bokeh.palettes import (Blues3, Blues4, Blues5,
                            Blues6, Blues7, Blues8, Blues9,
                            Greens3, Greens4, Greens5,
                            Greens6, Greens7, Greens8, Greens9,
                            Reds3, Reds4, Reds5, Reds6,
                            Reds7, Reds8, Reds9, Spectral3,
                            Spectral4, Spectral5, Spectral6,
                            Spectral7, Spectral8, Spectral9)
    if n <= 2:
        given_val = 3
    else:
        given_val = n
    if ptype=='magma':
        chosen = eval('Spectral' +'%s'%str(given_val))
    elif ptype == 'inferno':
        chosen = eval('Greens' + '%s'%str(given_val))
    elif ptype == 'plasma':
        chosen = eval('Reds' + '%s'%str(given_val))
    else:
        chosen = eval('Blues' + '%s'%str(given_val))
    if given_val == 3:
        return chosen[:n]
    return chosen


def grid_plot(plots, chunks=2):
    grid = gridplot(list(utils.chunks(plots, size=2)))
    show(grid)

def show_image(image, **kwargs):
    p = figure(x_range=(0, 10), y_range=(0, 10), **kwargs)
    p.image(image=[image], x=0, y=0, dw=10, dh=10, palette='Spectral11')
    return p

def show_var_imp(model, X,y):
    imp = pd.DataFrame(
        model.feature_importances_  ,
        columns = [ 'Importance' ] ,
        index = X.columns
        )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))


def lineplot(df, legend=None, title=None, **kwargs):
    assert all([ptypes.is_numeric_dtype(df[col]) for col in df.columns]), "Only numeric datatypes"
    if not title:
        title = "%s Vs %s" %(kwargs.get('xcol'), kwargs.get('ycol'))
    return Line(df, title=title,legend=True)

def histogram(histDF,values, bayesian_bins=False,**kwargs):
    if not bayesian_bins:
        return histDF[values].hist( **kwargs)
    else:
        import numpy as np
        bins = utils.bayesian_blocks(histDF[values])
        p1 = figure(title=kwargs.pop('title', 'Histogram of %s'%values),
                    tools="save")
        hist,edges = np.histogram(histDF[values], bins=bins)
        p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="#036564", line_color="#033649")
        p1.legend.location = "top_left"
        p1.xaxis.axis_label = 'x'
        p1.yaxis.axis_label = 'Frequency'
        return p1

def sb_barplot(df, category, column):
    return sns.barplot(x=category, y=column, data=df)


def heatmap(heatMapDF, xlabel, ylabel, value_label,
            title="heatmap", palette=None, width=500,
            height=500,**kwargs):
    from bokeh._legacy_charts import HeatMap
    if not palette:
        from bokeh.palettes import RdBu11 as palette_tmpl
        palette = palette_tmpl
    hm = HeatMap(heatMapDF, x=xlabel, y=ylabel, values=value_label,
                        title=title, height=height, width=width, palette=palette, **kwargs)
    return hm

def mtext(p, x, y, textstr, **kwargs):
   p.text(x, y, text=[textstr],
         text_color=kwargs.get('text_color'),
         text_align="center", text_font_size="10pt")

def sb_scatterplot(series1, series2):
    return sns.scatterplot(series1, series2)

def sb_histogram(series, kdeplot=False, bayesian_bins=False ):
    return sns.distplot(series, kde=kdeplot)

def sb_violinplot(series, dataframe=None, groupCol = None, **kwargs):
    if not groupCol:
        assert isinstance(series, pd.Series)
        plt = sns.violinplot(x=series, **kwargs)
    else:
        assert dataframe and groupCol
        assert isinstance(series, str)
        plt = sns.violinplot(x=groupCol, y=series, data=dataframe, **kwargs)
    return plt

def sb_jointplot(series1, series2):
    import numpy as np
    import seaborn as sns
    sns.set(style="white")

    # Show the joint distribution using kernel density estimation
    plt = sns.jointplot(series1, series2, kind="kde", size=7, space=0)
    return plt

def hyper_plot(dataframe, pca_plot=True, reduce_meth='SparsePCA', cluster=False, n_clusters=None, **kwargs):
    import hypertools as hyp
    if pca_plot:
        assert reduce_meth, 'must pass a reduction method'
        hyp.plot(dataframe, 'o', reduce=reduce_meth)
    elif cluster:
        assert n_clusters
        hyp.plot(dataframe, 'o', n_clusters=n_clusters, **kwargs)
    else:
        hyp.plot(dataframe, 'o', **kwargs)

