---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# Custom libraries
from datascienceutils import clusteringModels as cm
from datascienceutils import analyze
from datascienceutils import plotter
# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.charts import Histogram
import bokeh
output_notebook()

```

```python
irisDf = pd.read_csv('../data/Iris.csv')
```

```python
irisDf.head()
```

```python
target = irisDf.species
#irisDf.drop('Species', 1, inplace=True)
analyze.dimension_analyze(irisDf,group='species', animate=True, )
```

```python
plotter.hyper_plot(irisDf, cluster=True, n_clusters=3)
```

```python
plotter.hyper_plot(irisDf, pca_plot=True)
```

```python
cm.cluster_analyze(irisDf)
```

```python
cm.silhouette_analyze(irisDf, cluster_type='KMeans')
```

```python
cm.silhouette_analyze(irisDf, cluster_type='dbscan')
```

```python
cm.silhouette_analyze(irisDf, cluster_type='spectral')
```

```python
cm.silhouette_analyze(irisDf, cluster_type='birch')
```

```python
cm.som_analyze(irisDf, (10,10), algo_type='som')
```

```python
numColumns = irisDf.select_dtypes(include=[np.number])
cm.silhouette_analyze(numColumns, cluster_type='bgmm')
```
