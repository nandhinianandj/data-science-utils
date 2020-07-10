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
from dsu import analyze
from dsu import plotter
from dsu import sklearnUtils as sku

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
from sklearn import datasets
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
```

```python
CHECK_DISTS = ['norm', 'expon', 'logistic', 'cosine', 'cauchy',]

def distribution_tests(series, test_type='ks', dist_type=None):
    from scipy import stats
    if not dist_type:
        ks_test_results = pd.DataFrame(columns=['distribution', 'statistic', 'p-value'])
        for distribution in CHECK_DISTS:
            if test_type=='ks':
                print("Kolmogrov - Smirnov test with distribution %s"%distribution)
                try:
                    stat, pval = stats.kstest(series, distribution)
                except Exception:
                    print("Error for dist: %s"%distribution)
                    continue
                print(stat, pval)
                ks_test_results.append({'distribution': distribution,
                                        'statistic':stat,
                                        'p-value':pval},ignore_index=True)
            elif test_type =='wald':
                print("Wald test with distribution %s"%distribution)
                print(lm.wald_test(series, distribution))
            else:
                raise "Unknown distribution similarity test type"
        print(ks_test_results)
    else:
        print(stats.kstest(series, dist_type))
distribution_tests(diabetes_X[0])
```

```python
from dsu import statsutils as su
print(su.check_distribution(diabetes_X[0],))
```

```python
plotter.show(plotter.lineplot(pd.DataFrame(diabetes_X[0], columns=['0'])))
```

```python
from sklearn.externals import joblib

import os
import glob
files = glob.glob('../models/*.pkl')

print(files)
```

```python
from dsu import settings
settings.MODELS_BASE_PATH='../models'
models = dict()
for i,file in enumerate(files):
    model = file.split('/')[-1][:-4]
    models[i] = sku.load_model(model)
    
```

```python
models.items()
```

```python
res = list()
from dsu import modelValidators as mv
input_data = pd.DataFrame(diabetes_X_test)
input_data.columns = map(str, input_data.columns)
for k,(model, metadata) in models.items():
       
    res.append(model.predict(diabetes_X_test))
    model_info = metadata
    
    model_info.update({'model_class': 'regression'})
    print('model id: %s'%model_info['id'])
    
    if model_info['model_class'] == 'regression':
        # Check the predictions are type of continuous variables (float or int)
        # parse and translate output_metadata to choice of tests
        
        mv.validate(model, model_info, input_data) 
        
    

```
