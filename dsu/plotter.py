# Standard and External lib imports
from pandas.api import types as ptypes
from math import ceil


# plotting tools
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import plotnine

import itertools
import numpy as np
import operator
import os
import pandas as pd
import random



# TODO: Ugh.. this file/module needs a cleanup
# Custom imports
from . import utils

def init_and_get_figure(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    sns.set(style="white")

def sb_gridplot(df, col1, col2, category1, category2, chunks=2):
    if category:
        with sns.axes_style("white"):
            g = sns.FacetGrid(
                tips, row=category1, col=category2, margin_titles=True, height=2.5
            )
        g.map(plt.scatter, col1, col2, color="#334488", edgecolor="white", lw=0.5)
        g.set_axis_labels(col1, col2)
        # g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
        g.fig.subplots_adjust(wspace=0.02, hspace=0.02)


def sb_show():
    plt.show()


def show_image(image, **kwargs):
    p = plt.figure(x_range=(0, 10), y_range=(0, 10), **kwargs)
    p.image(image=[image], x=0, y=0, dw=10, dh=10, palette="Spectral11")
    return p


def lineplot(df, legend=None, title=None, **kwargs):
    assert all(
        [ptypes.is_numeric_dtype(df[col]) for col in df.columns]
    ), "Only numeric datatypes"
    if not title:
        title = "%s Vs %s" % (kwargs.get("xcol"), kwargs.get("ycol"))
    return Line(df, title=title, legend=True)


def mtext(p, x, y, textstr, **kwargs):
    p.text(
        x,
        y,
        text=[textstr],
        text_color=kwargs.get("text_color"),
        text_align="center",
        text_font_size="10pt",
    )


def sb_barplot(df, category, column, **kwargs):
    return sns.barplot(x=column, y=category, data=df, **kwargs)


def sb_heatmap(corr_df, **kwargs):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    return

def sb_scatterplot(series1, series2, **kwargs):
    return sns.scatterplot(series1, series2, **kwargs)


def sb_histogram(series, kdeplot=False, bayesian_bins=False):
    if kdeplot:
        return sns.distplot(series, kde=kdeplot)
    else:
        return sns.distplot(series, kde=kdeplot, kde_kws={"bw": "silverman"})


def sb_violinplot(series, dataframe=None, groupCol=None, **kwargs):
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


