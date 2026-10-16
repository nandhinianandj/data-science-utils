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
# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd
import random

from dsu import analyze
```

```python
irisDf = pd.read_csv('/home/nandhinianand/data/Iris.csv')
```

```python
numColumns = irisDf.select_dtypes(include=[np.number])
```

```python
analyze.regression_analyze(irisDf, numColumns, check_vif=False, check_heteroskedasticity=False)
```

```python
analyze.non_linear_regression_analyze(irisDf, 'sepal_length', 'sepal_width')
```

```python
analyze.pymc_regression_analyze(irisDf, numColumns)
```

```python
analyze.pymc_plot_posterior_glm(irisDf, 'sepal_length',  'sepal_width')
```

```python
analyze.pymc_log_reg_analyze(irisDf, numColumns)
```

```python
analyze.outliers_analyze(irisDf)
```

```python
```
