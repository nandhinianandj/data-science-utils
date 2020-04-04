---
jupyter:
  jupytext:
    formats: ipynb,md
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
import pandas as pd
import numpy as np
#import dask.DataFrame as dd
%matplotlib inline
import seaborn as sns
import json

#pd.set_option('display.max_colwidth', -1)
#pd.set_option('display.max_rows', -1)
```

```python tags=["parameters"]
fname = 'trip_context_data_v3.csv'
first_groupby='type'
second_groupby='ContextType'
third_groupby='triggerOrContextName'
fnam_prefix = fname.split('.')[0]
```

```python
dataframe = pd.read_csv(fname)
```

```python
new_df = pd.DataFrame(columns=['grouped_by', 'column_name', 'null_pct', 'min','mean','std', 'max'])
new_df['grouped_by'] = ['timeStamp' for x in range(len(dataframe.columns))]
new_df['column_name'] = dataframe.columns.tolist()

```

```python
numericalCols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
catCols = list(set(dataframe.columns) -set(numericalCols))

```

```python
dataframe.info()
```

## Find missing Values and their percent

```python
missing_dataframe = dataframe.isna()
```

```python
dataframe_missing_count = missing_dataframe.sum()
```

```python
print("NULL percent is files with context == Trip")
dataframe_missing_pct = dataframe_missing_count/len(dataframe)
dataframe_missing_pct
```

```python
zero_df = (dataframe == 0).sum(axis=0)
zero_df_pct = zero_df/len(dataframe)
zero_df_pct
```

```python
min_df = dataframe.min()
min_df
```

```python
max_df = dataframe.max()
max_df
```

```python
mean_df = dataframe.mean()
mean_df
```

```python
std_df = dataframe.std()
std_df
```

```python
max_df = dataframe.max()
max_df
```

```python
min_df
```

```python
#res_df=pd.concat([new_df, missing_dataframe,min_df, mean_df, std_df, max_df])
new_df['null_pct'] = dataframe_missing_pct.tolist()
def default_retrieve(df, col, default='test'):
    try:
        res = df[col]
    except KeyError:
        res = default
    return res

min_vals, max_vals, std_vals, mean_vals = list(), list(), list(), list()
for col in new_df.column_name:
    min_vals.append(default_retrieve(min_df, col, 'dummy_min'))
    max_vals.append(default_retrieve(max_df, col, 'dummy_max'))
    std_vals.append(default_retrieve(std_df, col, 'dummy_std'))
    mean_vals.append(default_retrieve(mean_df, col, 'dummy_mean'))


new_df['min'] = min_vals
new_df['max'] = max_vals
new_df['std'] = std_vals
new_df['mean'] = mean_vals
new_df['zeros_pct'] = zero_df_pct.values.tolist()
out_fname =  fnam_prefix + '_summary.csv'
new_df.to_csv(out_fname, index=False)

```

```python
all_null_cols = filter(lambda x: all(dataframe[x].isna()), dataframe.columns)
with open('all_null_columns.json', 'w+') as fd:
    json.dump(all_null_cols, fd)
#print(list(all_null_cols))
```

## Find descriptive stats grouped by categorical field values

```python
grouped_stats = dict()
```

```python
dataframe_trigger_grouped = dataframe.groupby(third_groupby)
dataframe_na_vals_cnt_grouped = dataframe_trigger_grouped.apply(lambda x: pd.isna(x)).count()

grouped_stats[third_groupby] = dataframe_na_vals_cnt_grouped/dataframe.groupby(third_groupby).count()
```

```python
grouped_stats
```

```python
dataframe_context_grouped = dataframe.groupby(second_groupby)
dataframe_na_vals_context_cnt_grouped = dataframe_context_grouped.apply(lambda x: pd.isna(x)).count()

grouped_stats[second_groupby] = dataframe_na_vals_context_cnt_grouped/dataframe.groupby(second_groupby).count()
```

```python
grouped_stats
```

```python
dataframe_type_grouped = dataframe.groupby(first_groupby)
dataframe_na_vals_type_cnt_grouped = dataframe_type_grouped.apply(lambda x: pd.isna(x)).count()

grouped_stats[first_groupby] = dataframe_na_vals_type_cnt_grouped/dataframe.groupby(first_groupby).count()
```

```python
grouped_stats
```

```python
for k,v in grouped_stats.items():
    grouped_stats[k].to_csv('%s_%s'%(fnam_prefix, 'grouped_by_' + str(k) + '.csv'), index=False)
```

```python
grouped_stats['triggerOrContextName']
```

```python

```
