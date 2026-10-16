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
from dsu import dimensionAnalyze as da
from dsu import utils
```

```python
# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd
import random
```

```python
irisDf = pd.read_csv('../data/train.csv')
numColumns = irisDf.select_dtypes(include=[np.number])
```

```python
# sample 10% of data
if len(irisDf) > 100000:
    samplesize =  int(0.1 *len(irisDf))
sampleDf = irisDf.sample(samplesize)
```

```python
da.tsne_dim_analyze(numColumns)
```

```python
da.hyperplot_analyze(irisDf,group='Class')
```

```python
da.fractal_analyze(irisDf, 'Class')
```

```python
for combo in utils.chunks(numColumns, size=2):
    da.fractal_dimension(irisDf[combo])
```

```python
da.hyperplot_analyze(irisDf)
```

```python
da.hyper_plot(irisDf, cluster=True, n_clusters=3)
```

```python
da.tsne_dim_analyze(irisDf)
```
