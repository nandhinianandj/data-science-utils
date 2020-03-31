# Standard and External lib imports
from pandas.api import types as ptypes
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

def sb_gridplot(df, col1, col2, category1, category2, chunks=2):
    if category:
        with sns.axes_style("white"):
            g = sns.FacetGrid(tips, row=category1, col=category2, margin_titles=True, height=2.5)
        g.map(plt.scatter, col1, col2, color="#334488", edgecolor="white", lw=.5);
        g.set_axis_labels(col1, col2 );
        #g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
        g.fig.subplots_adjust(wspace=.02, hspace=.02);

def sb_show():
    plt.show()

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

def mtext(p, x, y, textstr, **kwargs):
   p.text(x, y, text=[textstr],
         text_color=kwargs.get('text_color'),
         text_align="center", text_font_size="10pt")

def sb_barplot(df, category, column):
    return sns.barplot(x=category, y=column, data=df)

def sb_heatmap(corr_df, **kwargs):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return

def sb_scatterplot(series1, series2, **kwargs):
    return sns.scatterplot(series1, series2, **kwargs)

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

