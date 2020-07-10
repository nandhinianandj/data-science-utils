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
from dsu import analyze
from dsu import plotter

# Standard libraries
import datetime
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
irisDf = pd.read_csv('../data/Iris.csv')
```

```python
# sample 10% of data
if len(irisDf) > 100000:
    samplesize =  int(0.1 *len(irisDf))
sampleDf = irisDf.sample(samplesize)
```

```python
numericalCols = sampleDf.select_dtypes(include=[np.number]).columns
catCols = set(sampleDf.columns) -set(numericalCols)
```

```python
analyze.causal_analyze(sampleDf[numericalCols])
```

