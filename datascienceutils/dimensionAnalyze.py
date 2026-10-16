# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : dimensionAnalyze.py
#
#* Purpose :
#
#* Creation Date : 04-07-2020
#
#* Last Modified : Sunday 05 July 2020 04:14:27 AM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

from . import plotter
from . import utils

import pandas as pd
import numpy as np

def fractal_dimension(image, threshold=0.9):

    import scipy.misc
    import numpy as np
    # Only for 2d image
    assert len(image.shape) == 2
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(image, k):
        S = np.add.reduceat(
            np.add.reduceat(image, np.arange(0, image.shape[0], k), axis=0),
            np.arange(0, image.shape[1], k),
            axis=1,
        )

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform Z into a binary array
    image = image < threshold
    # Minimal dimension of image
    p = min(image.shape)
    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))
    # Extract the exponent
    n = int(np.log(n) / np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(image, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def fractal_analyze(dataframe, column, L=None, dim_type="box"):
    if dim_type == "box":
        plotter.sb_show(_box_dimension(dataframe, column, L=L))
    else:
        plotter.sb_show(_hausdorff_dimension(dataframe[column], L=L))

def hyperplot_analyze(dataframe, group=None, **kwargs):
    # Add pca, svd, linear discriminant analysis, tsne
    if not group:
        hyper_plot(dataframe, **kwargs)
    else:
        groupVals = dataframe[group]
        groupLabels = dataframe[group].unique()
        dataframe.drop(group, 1, inplace=True)
        hyper_plot(dataframe, group=groupVals, legend=groupLabels, **kwargs)
    pass


def tsne_dim_analyze(dataframe, datatype="binary", **kwargs):
    from MulticoreTSNE import MulticoreTSNE as TSNE

    n_jobs = kwargs.get("n_jobs", 4)
    tsne = TSNE(n_jobs=n_jobs)
    Y = tsne.fit_transform(dataframe)
    df = pd.DataFrame()
    df["x"] = Y[0, 0]
    df["y"] = Y[0, 1]
    plotter.sb_scatterplot(df["x"], df["y"])


def fractaldim(pointlist, boxlevel):
    """Returns the approximate fractal dimension of pointlist,
    via the box-counting algorithm.  The elements of pointset
    should be three-element sequences of numbers between 0.0
    and 1.0.  The boxlevel is the number of divisions made on
    each dimension, and should be greater than 1."""
    # From here https://mail.python.org/pipermail/python-list/2000-September/025263.html

    if boxlevel <= 1.0:
        return -1

    pointdict = {}

    def mapfunction(val, level=boxlevel):
        return int(val * level)

    for point in pointlist:
        box = (
            int(point[0] * boxlevel),
            int(point[1] * boxlevel),
            int(point[2] * boxlevel),
        )
        # box = tuple(map(mapfunction, point))
        if not pointdict.has_key(box):
            pointdict[box] = 1

    num = len(pointdict.keys())

    if num == 0:
        return -1

    return math.log(num) / math.log(boxlevel)

###########################################
### UTILS for above functions
###########################################

def _hausdorff_dimension(pixels):

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(1, 8, num=20, endpoint=False, base=2)
    Ns = []
    # looping over several scales
    for scale in scales:
        print("Scale ", scale)
        # computing the histogram
        H, edges = np.histogramdd(
            pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale))
        )
        Ns.append(np.sum(H > 0))

    plot_df = pd.DataFrame(columns=["log(scales)", "log(Ns)"])
    plot_df["Log(scales)"] = np.log(scales)
    plot_df["Log(Ns)"] = np.log(Ns)
    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    plotter.sb_lineplot('Log(scales)', 'Log(Ns)', plot_df,
                            title="log(scales) Vs log(Ns)")

def _box_dimension(dataframe, column, L=None):
    if not L:
        L = dataframe[column].max()
    r = np.array([L / (2.0 ** i) for i in range(12, 0, -1)])
    N = [utils.count_boxes(dataframe[column], ri, L) for ri in r]

    def f(x, A, Df):
        """
        User defined function for scipy.optimize.curve_fit(),
        which will find optimal values for A and Df.
        """
        return Df * x + A

    import scipy

    popt, pcov = scipy.optimize.curve_fit(f, np.log(1.0 / r), np.log(N))
    A, Df = popt
    new_df = pd.DataFrame(columns=["Box Size(1/r)", "No. of Boxes"])
    new_df["Box Size(1/r)"] = 1 / r
    new_df["No. of Boxes"] = N
    return plotter.sb_lineplot("Box Size(1/r)", "No. of Boxes", new_df, title="Box Size(1/r) Vs No. of Boxes")

def count_boxes(data, box_size, M):
    # data = Series( data )
    N = np.int(np.floor(M / box_size))
    counts = list()
    for i in range(N):
        condition = (data >= i * box_size) & (data < (i + 1) * box_size)
        subset = data[condition]
        counts.append(subset.count())
    counts = [i for i in counts if i != 0]
    return len(counts)

def hyper_plot( dataframe, pca_plot=True, reduce_meth="SparsePCA",
                cluster=False, n_clusters=None, **kwargs):
    import hypertools as hyp

    if pca_plot:
        assert reduce_meth, "must pass a reduction method"
        hyp.plot(dataframe, "o", reduce=reduce_meth)
    elif cluster:
        assert n_clusters
        hyp.plot(dataframe, "o", n_clusters=n_clusters, **kwargs)
    else:
        hyp.plot(dataframe, "o", **kwargs)
