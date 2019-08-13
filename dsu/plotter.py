# Standard and External lib imports
from pandas.api import types as ptypes
from bokeh.mpl import to_bokeh
from bokeh.io import gridplot
from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import ( Text, PanTool, WheelZoomTool, LinearAxis,
                           SingleIntervalTicker, Range1d,  Plot,
                           Text, Circle, HoverTool, Triangle)
from bokeh.charts import Chart, Line
from math import ceil

import matplotlib
from matplotlib import pyplot as plt
import itertools
import numpy as np
import operator
import os
import random
import tempfile

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

def contour_plot(dataframe, model , **kwargs):
    import matplotlib.pyplot as plt
    xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if not levels:
        levels = np.linspace(Z.min(), threshold, 7)
    plot1 = plt.plot()
    plot1.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.Blues_r, **kwargs)
    return plot1

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

def show_graph(graph):
    import networkx
    fout = tempfile.NamedTemporaryFile(suffix='.png')
    dot_fname = '.'.join([fout.name.split('.')[0], 'dot'])
    gr = networkx.draw_graphviz(graph)
    dot_data = tree.export_graphviz(model, out_file=dot_fname)
    os.system('dot -Tpng %s -o %s'%(dot_fname, fout.name))
    show(show_image(io.imread(fout.name)))
    os.remove(dot_fname)

def show_tree_model(model, model_type='tree'):
    assert model_type in ['tree', 'randomforest', 'xgboost']
    from sklearn import tree
    import pydotplus
    from skimage import io
    #assert isinstance(model, tree.DecisionTreeClassifier)
    if model_type == 'tree':
        fout = tempfile.NamedTemporaryFile(suffix='.png')
        dot_fname = '.'.join([fout.name.split('.')[0], 'dot'])
        dot_data = tree.export_graphviz(model, out_file=dot_fname)
        os.system('dot -Tpng %s -o %s'%(dot_fname, fout.name))
        show(show_image(io.imread(fout.name)))
        os.remove(dot_fname)
    elif model_type == 'randomforest':
        graph_plots = list()
        if len(model.estimators_) > 10:
            print("Sorry more that 10 trees can't be displayed")
            return
        for tree_model in model.estimators_:
            fout = tempfile.NamedTemporaryFile(suffix='.png')
            dot_fname = '.'.join([fout.name.split('.')[0], 'dot'])
            dot_data = tree.export_graphviz(tree_model, out_file=dot_fname)
            os.system('dot -Tpng %s -o %s'%(dot_fname, fout.name))
            graph_plots.append(show_image(io.imread(fout.name)))
        grid = gridplot(list(utils.chunks(graph_plots, size=3)))
        show(grid)
        os.remove(dot_fname)
    else:
        #It must be xgboost
        import xgboost
        xgboost.to_graphviz(model)
        fout = tempfile.NamedTemporaryFile(suffix='.png')
        dot_fname = '.'.join([fout.name.split('.')[0], 'dot'])
        dot_data = tree.export_graphviz(tree_model, out_file=dot_fname)
        os.system('dot -Tpng %s -o %s'%(dot_fname, fout.name))
        show(show_image(io.imread(fout.name)))
        os.remove(dot_fname)

def show_model_interpretation(model, model_type='randomforest'):
    #TODO: Use lime
    import lime
    pass

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

def barplot(barDF, xlabel, ylabel, title="Bar Plot",
                            agg='sum', **kwargs):
    barplot = barDF.bar(x=xlabel, y=ylabel, agg=agg, title=title, **kwargs)
    return barplot

def boxplot(boxDF, values_label, xlabel, title="boxplot", **kwargs):
    from bokeh.charts import BoxPlot
    boxplot = BoxPlot(boxDF, values=values_label, label=xlabel, color=xlabel, title=title, **kwargs)
    return boxplot

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

def scatterplot(scatterDF, xcol, ycol,
                xlabel=None, ylabel=None,
                groupCol=None, plttitle=None, **kwargs):
    fig_kwargs = kwargs.get('figure')
    if fig_kwargs:
        p = figure(title=plttitle, **fig_kwargs)
    else:
        p = figure(title=plttitle)
    from bokeh.charts import Scatter

    if not xlabel:
        xlabel = xcol
    if not ylabel:
        ylabel = ycol

    if not groupCol:
        kwargs.pop('groupCol', None)
        p.circle(scatterDF[xcol], scatterDF[ycol], size=5, **kwargs)
    else:
        groups = list(scatterDF[groupCol].unique())
        colors = genColors(len(groups), ptype='plasma')
        colors = list(np.hstack([colors] * 20))
        for group in groups:
            color = colors.pop(random.randrange(len(colors)))
            p.circle(scatterDF[scatterDF[groupCol]==group][xcol],
                     scatterDF[scatterDF[groupCol]==group][ycol],
                        size=5, color=color )
    p.xaxis.axis_label = str(xcol)
    p.yaxis.axis_label = str(ycol)
    return p

def pieChart(df, column, **kwargs):

    wedges = []
    wedge_sum = 0
    total = len(df)
    colors = genColors(df[column].nunique(), ptype='magma')
    for i, (key, val) in enumerate(df.groupby(column).size().iteritems()):
        wedge = dict()
        pct = val/float(total)
        wedge['start'] = 2 * np.pi * wedge_sum
        wedge_sum = (val/float(total)) + wedge_sum
        wedge['end'] = 2 * np.pi * wedge_sum
        wedge['name'] = '{}-{:.2f} %'.format(key, pct)
        wedge['color'] = colors.pop()
        wedges.append(wedge)
    p = figure(x_range=(-1,1), y_range=(-1,1), x_axis_label=column, **kwargs)

    for i, wedge in enumerate(wedges):
        p.wedge(x=0, y=0, radius=1, start_angle=wedge['start'], end_angle=wedge['end'],
                color=wedge['color'], legend=wedge['name'])
    return p

def mcircle(p, x, y, **kwargs):
    p.circle(x, y, **kwargs)

def mscatter(p, x, y, typestr="o"):
    p.scatter(x, y, marker=typestr, alpha=0.5)

def mtext(p, x, y, textstr, **kwargs):
   p.text(x, y, text=[textstr],
         text_color=kwargs.get('text_color'),
         text_align="center", text_font_size="10pt")

def boxplot(xrange, yrange, boxSource, xlabel='x', ylabel='y', title='title', colors=list()):
    p=figure(
        title=title,
        x_range=xrange,
        y_range=yrange)
        #tools=TOOLS)

    p.plot_width=900
    p.plot_height = 400
    p.toolbar_location='left'

    p.rect(xlabel, ylabel, 1, 1, source=boxSource, color=colors, line_color=None)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi/3

    return p

def sb_violinplot(series, dataframe=None, groupCol = None, **kwargs):
    import seaborn as sns
    sns.set(style='white')
    import pandas as pd
    if not groupCol:
        assert isinstance(series, pd.Series)
        plt = sns.violinplot(x=series, **kwargs)
    else:
        assert dataframe and groupCol
        assert isinstance(series, str)
        plt = sns.violinplot(x=groupCol, y=series, data=dataframe, **kwargs)
    plt.show()

def sb_jointplot(series1, series2):
    import numpy as np
    import seaborn as sns
    sns.set(style="white")

    # Show the joint distribution using kernel density estimation
    plt = sns.jointplot(series1, series2, kind="kde", size=7, space=0)

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

def gp_pointplot(geo_dataframe, geo_locations, scale_column):
    import geoplot.crs as gcrs
    import geoplot as gplt

    proj = gcrs.AlbersEqualArea()# central_longitude=-98, central_latitude=39.5)

    ax = gplt.polyplot(geo_locations,
                       projection=proj,
                       zorder=-1,
                       linewidth=0.5,
                        legend_kwargs={'frameon': False, 'loc': 'lower right'},
                       **kwargs
                       )
    gplt.pointplot(geo_dataframe,
                   scale=scale_column,
                   ax=ax,
                   projection=proj,
                   **kwargs
                   )
    pass

def show_volume(dataframe, cols, vec_cols=None, plt_type='vol'):
    import ipyvolume as ipv
    if plt_type=='vol':
        assert len(cols) == 3, "Only 3d Volumes"
        ipv.quickvolshow(dataframe[cols], level=[0.25, 0.75], opacity=0.03, level_width=0.1, data_min=0,
                data_max=1)
    elif plt_type=='scatter':
        ipv.quickscatter(dataframe[cols], size=1, marker='sphere')

    elif plt_type=='quiver':
        assert vec_cols, 'Vector Columns needed'
        x,y,z = dataframe[cols]
        u,v,w = dataframe[vec_cols]
        ipv.quickquiver(x,y,z, u,v,w, size=5)
    elif plt_type('mesh'):
        m = ipv.plot(x,y,z,wireframe=False)
        ipv.squarelim()
        ipv.show()
    else:
        raise 'Unsupported plot type'
