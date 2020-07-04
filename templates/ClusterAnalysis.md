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
from dsu import clusteringModels as cm
from dsu import plotter
# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd
```

```python
irisDf = pd.read_csv('../data/Iris.csv')
```

```python
irisDf.head()
```

```python
# sample 10% of data
if len(irisDf) > 100000:
    samplesize =  int(0.1 *len(irisDf))
sampleDf = irisDf.sample(samplesize)
```

```python
target = irisDf.species
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
