## TODO
	* Add model validation syntax and make it mandatory while dumping a model
	* Also add some code that reads the validation parameters and generates test cases to test
	  /validate the model(stats tests, input distributions, output distributions etc..)

        * Add a contour plot

	* Yellow-fin algorithm(http://cs.stanford.edu/~zjian/project/YellowFin/)

	* Add test of independence support for categorical variables(use that P(A & B) = P(A) * P(B)
	  as a test

	* Add a framework for defining the model's limits and trigger alerts, when it is
	  violated.(use pymc/pystan for creating conditions about probability distributions of data
	  sources.)

	* Add pymc based linear regression(https://github.com/pymc-devs/pymc/wiki/StraightLineFit)
	  and this (http://pymc-devs.github.io/pymc/tutorial.html#an-example-statistical-model) and
	  pystan (https://pystan.readthedocs.io/en/latest/)

	* Add a wrapper for geo-plot in plotter.py(for plotting on a map)

	* Add support for feature filtering..(tsfresh module and also others) in features.py

	* Add Gini Coefficient-like measure visual for the cluster analyze

	* Add dendrogram style visuals for cluster analyze

	* Add support for https://github.com/ANNetGPGPU/ANNetGPGPU in the cluster analyze logic

	* Add [Permute](https://github.com/statlab/permute) tests

	* Finish off that kernel density estimation for 1-d data clustering(http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html) and (http://bokeh.pydata.org/en/0.9.3/docs/reference/charts.html).

	* Add [Fractal Analysis](https://en.wikipedia.org/wiki/Fractal_analysis) R-equivalent
	  package[fdim](http://cran.r-project.org/src/contrib/Archive/fdim/)

	* Add [Renormalization group analysis](https://medium.com/incerto/the-most-intolerant-wins-the-dictatorship-of-the-small-minority-3f1f83ce4e15)

	* Add a dimensional analysis function/option.
		* - No. of independent(aka orthogonal) dimensions(aka degrees of freedom of the
		                  system)
		* - Maximally orthogonal combinations of all given features/dimensions.
		* - Most significant dimensions(w.r.t predicted/target variable)
		* - Also add interpretation of factor_analyze.(the transformations pca/lda acts on
		                  the dataset)
		* - http://www.kdnuggets.com/2015/05/7-methods-data-dimensionality-reduction.html
		* - https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/

	* Add [Correspondence
	  Analysis](http://www.mathematica-journal.com/2010/09/an-introduction-to-correspondence-analysis/)

	* Sensitivity analysis

	* Interaction Analysis

	* Add 3D heatmaps and may be 3D + 1D(time) visualizations/Animations(like the gapminder
	  bubble chart for ex:)

	* Add a dask/airflow/luigi/pinball support for training models with different samples on
	  distributed systems.

	* Add a separate grid search function to grid search a data set with the given
	  model.(wrapper around sklearn model_selection's grid search)

	* Add Gaussian Mixture Model to clustering models

	* Add plots for regression analysis with different models(may be [r-squared like]
	  http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
	  or somethin else

	* Add a way to check for non-linear correlations(aka ace algorithm)

	* Implement the trellis plots for correlation analyze (when there's categories)

	* Setup python sphinx and add proper documentation for all classes and functions

	* Create a function to take dataframe, run tree/randomforest, pick out best tree, create a
	  neural network based on the tree, and return it.. (The user can then train it).

	*[Agglomerative Hierarchical clustering](https://www.xlstat.com/en/solutions/features/agglomerative-hierarchical-clustering-ahc)

	* ~~Show off that bayesian binning in dist_analyze plots~~

	* ~~ Add support for [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) ~~

	* ~~ dist_analyze TODO: May be add a way to plot joint distributions of two variables? ~~

	* ~~Cleanup/refactor the plotter.py to remove obsolete/unused plots~~

	* ~~Refactor out matplotlib dependencies present in clusteringModels and predictiveModels
	  modules, replacing with calls to the plotter module(which wraps bokeh and seaborn)~~

	* ~~dist_analyze TODO: add grouped violinplots by categorical variables too.~~

	* ~~Add factor_analyze function to analyze.py(probably something like PCA or the likes)~~

	* ~~ Show pie-charts for dist_analyze of categories only when there's < 5 categories. else use
	  horizontal stacked bar charts. ~~


