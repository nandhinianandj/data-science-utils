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

import seaborn as sns

# Standard libraries
import json
%matplotlib inline
import datetime
import numpy as np
import pandas as pd
import random

import pandas_profiling as pp
import pprint

from matplotlib import pyplot as plt
```

```python tags=["parameters"]
fname = '../data/trip_context_data.csv'
```

```python
df = pd.read_csv(fname)

numericalCols = df.select_dtypes(include=[np.number]).columns.tolist()
catCols = list(set(df.columns) -set(numericalCols))
catCols.remove('timestamp')
catCols
```

```python
df['ContextType'].unique()
```

```python
df.describe()
```

```python

for col in numericalCols:
    sns.violinplot(x=col, y='type', data=df, inner=None)
    #plt.show()
    sns.swarmplot(x=col, y='type', data=df, color="white", edgecolor="gray")
    plt.show()
```

```python

for col in numericalCols:
    sns.violinplot(x=col, y='ContextType', data=df, inner=None)
    #plt.show()
    sns.swarmplot(x=col, y='ContextType', data=df, color="white", edgecolor="gray")
    plt.show()
```

```python

```
