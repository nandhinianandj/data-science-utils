## TODO
	* TSNE based dimensionalty analysis [function.](https://github.com/DmitryUlyanov/Multicore-TSNE)
	* Topological Data
	  Analysis(https://gist.github.com/anandjeyahar/df6d477271cf49b152f630aa72bc27c3)
	* Consider adding [swiftapply](https://github.com/jmcarpenter2/swifter) whereever you use pandas
	* Add [Metric learning methods](https://github.com/metric-learn/metric-learn)
	* [Pystacknet](https://github.com/h2oai/pystacknet) integration for support for integrated
	  models.
	* [partial path demo](http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html) for tree-based mmethods(random-forest, gradient boosted etc..)
	* [Decision Tree visualization](http://explained.ai/decision-tree-viz/index.html) or
	  [this](https://github.com/parrt/animl/tree/master)

	* [Causality analysis](https://www.arxiv-vanity.com/papers/1809.09337/)
	* Also add some code that reads the validation parameters and generates test cases to test
	  /validate the model(stats tests, input distributions, output distributions etc..)

	* Yellow-fin algorithm(http://cs.stanford.edu/~zjian/project/YellowFin/)

	* Add test of independence support for categorical variables(use that P(A & B) = P(A) * P(B)
	  as a test

	* Add a framework for defining the model's limits and trigger alerts, when it is
	  violated.(use pymc/pystan for creating conditions about probability distributions of data
	  sources.)

	* Add support for https://github.com/ANNetGPGPU/ANNetGPGPU in the cluster analyze logic

	* Finish off that kernel density estimation for 1-d data clustering(http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html) and (http://bokeh.pydata.org/en/0.9.3/docs/reference/charts.html).

	* Add [Fractal Analysis](https://en.wikipedia.org/wiki/Fractal_analysis) R-equivalent
	  package[fdim](http://cran.r-project.org/src/contrib/Archive/fdim/) a python [oss project](https://github.com/kaziida24/fractal)

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

	* [Sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis)

	* Interaction Analysis

	* [Survival
	  Analysis](https://www.nature.com/bjc/journal/v89/n2/pdf/6601118a.pdf?foxtrotcallback=true)

	* Add a separate grid search function to grid search a data set with the given
	  model.(wrapper around sklearn model_selection's grid search)

	* Add Gaussian Mixture Model to clustering models

	* Add plots for regression analysis with different models(may be [r-squared like]
	  http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
	  or somethin else

	* Add a way to check for non-linear correlations(aka ace algorithm)


	* Create a function to take dataframe, run tree/randomforest, pick out best tree, create a
	  neural network based on the tree, and return it.. (The user can then train it).

	* [Agglomerative Hierarchical clustering](https://www.xlstat.com/en/solutions/features/agglomerative-hierarchical-clustering-ahc)

