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
from datascienceutils import plotter
from datascienceutils import analyze

# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd
import random


from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.charts import Histogram
import bokeh
output_notebook()

```

```python
irisDf = pd.read_csv('../data/Iris.csv')
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

```
analyze.missing_values(df, sample=True, samplesize=1000)
```

```python
import itertools
for combo in itertools.combinations(numericalCols, 2):
    analyze.correlation_analyze(irisDf, combo[0], combo[1],
                                measures=['sepal_length', 'petal_length', 'petal_width', 'sepal_width'])
```

```python
for col in numericalCols:
    plotter.show(analyze.dist_analyze(irisDf, col, kdeplot=True, bayesian_hist=True))
for col in catCols:
    plotter.show(analyze.dist_analyze(irisDf, col, kdeplot=False, bayesian_hist=False))
```

```python
#irisDf.drop('filename', inplace=True, axis=1)
for col in numericalCols:
    plotter.show(analyze.dist_analyze(irisDf, col, bayesian_hist=True))
```

```python
analyze.regression_analyze(irisDf, numericalCols, check_vif=False, check_heteroskedasticity=False, check_dist_similarity=False)
```

```python
analyze.non_linear_regression_analyze(irisDf, numericalCols, check_vif=False, check_heteroskedasticity=False)
```

```python
analyze.bayesian_regression_analyze(irisDf, numericalCols)
```

```python
irisDf.head()
```

```python
irisDf.tail()
```

```python
irisDf['Class'] = [random.choice(range(2)) for x in range(len(irisDf))]
irisDf.head()
analyze.dimension_analyze(irisDf,group='Class')
```
