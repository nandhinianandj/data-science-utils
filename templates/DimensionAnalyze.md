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
from datascienceutils import analyze
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
irisDf = pd.read_csv('/home/anand/DataScientist/data/Iris.csv')
numColumns = irisDf.select_dtypes(include=[np.number])
```

```python
analyze.tsne_dim_analyze(numColumns)
```

```python
irisDf['Class'] = [random.choice(range(2)) for x in range(len(irisDf))]
irisDf.head()
analyze.hyperplot_analyze(irisDf,group='Class')
```

```python
for col in numColumns:
    analyze.fractal_dimension(irisDf, col)
```

```python

```
