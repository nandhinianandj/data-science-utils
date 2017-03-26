from bokeh.layouts import gridplot
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph, KernelDensity
import numpy as np
import pandas as pd
import time

# Custom utils
from . import sklearnUtils as sku
from . import plotter
from . import utils

#TODO: add a way of weakening the discovered cluster structure and running again
# http://scilogs.spektrum.de/hlf/sometimes-noise-signals/
def is_cluster(dataframe, model_type='dbscan', batch_size=2):
    if model_type == 'dbscan':
        model_obj = cluster.DBSCAN(eps=.2)
    elif model_type == 'MiniBatchKMeans':
        assert batch_size, "Batch size mandatory"
        model_obj = cluster.MiniBatchKMeans(n_clusters=batch_size)
    else:
        pass
    model_obj.fit(X)
    return model_obj.cluster_centers_

def cluster_analyze(dataframe):
    plots = list()
    if len(dataframe.columns) > 1:
        clustering_names = [
        'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
        'SpectralClustering', 'Ward', 'AgglomerativeClustering',
        'DBSCAN', 'Birch']
        # normalize dataset for easier parameter selection
        X = sku.feature_scale_or_normalize(dataframe, dataframe.columns)
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # create clustering estimators
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=2)
        ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                               connectivity=connectivity)
        spectral = cluster.SpectralClustering(n_clusters=2,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=.2)
        affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                           preference=-200)

        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock", n_clusters=2,
            connectivity=connectivity)


        birch = cluster.Birch(n_clusters=2)
        clustering_algorithms = [
            two_means, affinity_propagation, ms, spectral, ward, average_linkage,
            dbscan, birch]
    	for name, algorithm in zip(clustering_names, clustering_algorithms):
    	    # predict cluster memberships
    	    t0 = time.time()
    	    algorithm.fit(X)
    	    t1 = time.time()
    	    if hasattr(algorithm, 'labels_'):
    	        print("According to %s there are %d clusters"%(name, len(set(algorithm.labels_))))
    	        y_pred = algorithm.labels_.astype(np.int)
    	    else:
    	        y_pred = algorithm.predict(X)

    	    # plot
    	    plot_data = np.c_[X, y_pred]
    	    columns = list(dataframe.columns) + ['classes']
    	    new_df = pd.DataFrame(data=plot_data, columns=columns)
    	    s_plot = plotter.scatterplot(new_df, columns[0], columns[1], plttitle='%s'%name, groupCol='classes')
    	    plots.append(s_plot)

    	    if hasattr(algorithm, 'cluster_centers_'):
    	        print("According to %s there are %d clusters"%(name, len(algorithm.cluster_centers_)))
    	        centers = pd.DataFrame(algorithm.cluster_centers_)
    	        for i, c in enumerate(algorithm.cluster_centers_):
    	            # Draw white circles at cluster centers
    	            plotter.mtext(s_plot, c[0], c[1], "%s"%str(i), text_color="red")

    else:
        X = sku.feature_scale_or_normalize(dataframe, dataframe.columns)
	for kernel in ['gaussian', 'tophat', 'epanechnikov']:
	    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
	    log_dens = kde.score_samples(X_plot)
            hz = Horizon(xyvalues, index='Date', title="Horizon Example", ylabel='Sample Data', xlabel='')
	    ax.plot(X[:, 0], np.exp(log_dens), '-',
	            label="kernel = '{0}'".format(kernel))

    grid = gridplot(list(utils.chunks(plots,size=2)))
    plotter.show(grid)


def silhouette_analyze(dataframe, cluster_type='KMeans', n_clusters=None):
    """
    Plot silhouette analysis plot of given data and cluster type across different  cluster sizes
    # from here Silhouette analysis --http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """
    # Use clustering algorithms from here
    # http://scikit-learn.org/stable/modules/clustering.html#clustering
    # And add a plot that visually plotter.shows the effectiveness of the clusters/clustering rule.(may be
    # coloured area plots ??)
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.cm as cm
    import numpy as np
    import collections
    if not n_clusters:
        n_clusters = range(2, 8, 2)
    assert isinstance(n_clusters, collections.Iterable), "n_clusters must be an iterable object"
    dataframe = dataframe.as_matrix()
    cluster_scores_df = pd.DataFrame(columns=['cluster_size', 'silhouette_score'])
    for j, cluster in enumerate(n_clusters):
        clusterer = utils.get_model_obj(cluster_type, n_clusters=cluster)

        # Initialize the clusterer with n_clusters value and a random generator
        cluster_labels = clusterer.fit_predict(dataframe)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(dataframe, cluster_labels)
            cluster_scores_df.loc[j] = [cluster, silhouette_avg]
            print("For clusters =", cluster,
                    "The average silhouette_score is :", silhouette_avg)
        else:
            print("No cluster found with cluster no:%d and algo type: %s"%(cluster, cluster_type))
            continue

        # 2nd Plot showing the actual clusters formed
        dataframe = pd.DataFrame(dataframe)
        cols = list(dataframe.columns)
        dataframe['predictions'] = pd.Series(cluster_labels)
        s_plot = plotter.scatterplot(dataframe,
                                     cols[0], cols[1],
                                     groupCol='predictions',
                                     xlabel="Feature space for 1st feature",
                                     ylabel="Feature space for 2nd feature",
                                     plttitle="Visualization of the clustered data")

        if hasattr(clusterer, 'cluster_centers_'):
            # Labeling the clusters
            centers = pd.DataFrame(clusterer.cluster_centers_)
            for i, c in enumerate(clusterer.cluster_centers_):
                # Draw white circles at cluster centers
                plotter.mtext(s_plot, c[0], c[1], "%s"%str(i), text_color="red")
        plotter.show(s_plot)

    plotter.show(plotter.lineplot(cluster_scores_df, xcol='cluster_size', ycol='silhouette_score'))

def som_analyze(dataframe, mapsize, algo_type='som'):
    import sompy
    som_factory = sompy.SOMFactory()
    data = dataframe.as_matrix()
    assert isinstance(mapsize, tuple), "Mapsize must be a tuple"
    sm = som_factory.build(data, mapsize= mapsize, normalization='var', initialization='pca')
    if algo_type == 'som':
        sm.train(n_job=6, shared_memory='no', verbose='INFO')

        # View map
        v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
        v.show(sm, what='codebook', cmap='jet', col_sz=6) #which_dim=[0,1]
        v.show(sm, what='cluster', cmap='jet', col_sz=6) #which_dim=[0,1] defaults to 'all',

        # Hitmap
        h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
        h.show(sm)

    elif algo_type == 'umatrix':
        #But Umatrix finds the clusters easily
        u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
        #This is the Umat value
        UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)
        u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=False, blob=False)
    else:
        raise "Unknown SOM algorithm type"

