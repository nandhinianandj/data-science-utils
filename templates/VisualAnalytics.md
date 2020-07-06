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
from dsu import plotter
from dsu import analyze

# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd
import random

```

```python
irisDf = pd.read_csv('~/playspace/data/Iris.csv')
```

```python
numericalCols = irisDf.select_dtypes(include=[np.number]).columns
catCols = set(irisDf.columns) -set(numericalCols)
```

```python
irisDf.describe()
```

```python
irisDf.head()
```

```python
irisDf.var()
```

```python
irisDf.skew()
```

```python
irisDf.corr()
```

```python
# sample 10% of data
if len(irisDf) > 100000:
    samplesize =  int(0.1 *len(irisDf))
    sampleDf = irisDf.sample(samplesize)
else:
    sampleDf = irisDf
```

```python
analyze.missing_values(sampleDf,)
```

```python
analyze.missing_values(sampleDf, groups = ['Class'])
```

```python
import itertools
for combo in itertools.combinations(numericalCols, 2):
    analyze.correlation_analyze(sampleDf, combo[0], combo[1],
                                measures=['sepal_length', 'petal_length', 'petal_width', 'sepal_width'],
				check_linearity=True, trellis=True)
```

```python
for col in numericalCols:
    for catCol in catCols:
        analyze.dist_analyze(sampleDf, col, category=catCol, kdeplot=True, bayesian_hist=True)

```

```python
for catCol in catCols:
    for col in numericalCols:
        print(col, catCol)
        plotter.sb_violinplot(col, groupCol=catCol, dataframe=sampleDf)
        plotter.sb_show()
        
```


```python
analyze.joint_dist_analyze(sampleDf, columns=numericalCols, categories=catCols)
```

