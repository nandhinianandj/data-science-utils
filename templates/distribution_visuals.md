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

```python
metadataDf = pd.read_csv('../data/Finalized data parameters_charging.csv')
metadataDf.head()

renames=dict(zip(metadataDf['Gen2/2K'].apply(lambda x: x.lower().replace('.', '_').replace(',','')).tolist(), 
                metadataDf['Unnamed: 1'].apply(lambda x: x.strip().replace(' ', '_').replace(',', '').replace('.', '_')).tolist()))

```

```python
df = pd.read_csv('../data/gen2k_chd_sample.csv')
df.rename(columns=renames, inplace=True)
df.rename(columns={'charging_history__charge_type__100V_system_200V_systemRapid': 'charge_type'}, inplace=True)

numericalCols = df.select_dtypes(include=[np.number]).columns.tolist()
catCols = list(set(df.columns) -set(numericalCols))
catCols
```

```python
df.head()
```

```python
df.describe()
```

```python
#pp.ProfileReport(df)
```

```python
df.drop(df[df['charging_history__charging_start_date_and_time___month']==0].index, inplace=True)
df.drop(df[df['charging_history__Charge_termination_date_and_time___month']==0].index, inplace=True)

#mod_df.charging_history_charge_start_time_year.fillna(value=2020, inplace=True)
#mod_df.charging_history_charge_term_time_year.fillna(value=2020, inplace=True)

#mod_df.charging_history_charge_start_time_month.fillna(value=1, inplace=True)
#mod_df.charging_history_charge_term_time_month.fillna(value=1, inplace=True)


df['charging_history__charging_start_date___the_day'].fillna(value=1, inplace=True)
df['charging_history__Charge_termination_date_and_time___the_day'].fillna(value=1, inplace=True)
#any(irisDf.charging_history_charge_start_time_day.isna())

df['charging_history__charging_start_time_date_and_time__'].fillna(value=1, inplace=True)
df['charging_history__When_charging_the_end_date_and_time__'].fillna(value=1, inplace=True)
df['charging_history___charging_start_date_and_time___minute'].fillna(value=1, inplace=True)
df['charging_history__Charge_termination_date_and_time____minutes'].fillna(value=1, inplace=True)

df['charging_history__charging_start_date___the_day'] = mod_df.loc[mod_df.charging_history_charge_start_time_month==2,
        'charging_history_charge_start_time_day'].replace(to_replace=[29, 30,31], value=[28, 28, 28])

mod_df.charging_history_charge_term_time_day = mod_df.loc[mod_df.charging_history_charge_term_time_month==2,
        'charging_history_charge_term_time_day'].replace(to_replace=[29, 30, 31], value=[28,28,28])
```

```python

for col in numericalCols:
    sns.violinplot(x=col, y='charge_type', data=df, inner=None)
    #plt.show()
    #sns.swarmplot(x=col, y='charge_type', data=df, color="white", edgecolor="gray")
    plt.show()
```

## Few observations
  ### Obvious(atleast in hindsight) ones.
     * -- charge_type has almost zero impact/effect on the charging_end_time_IR_sensor_value or charging_history
     * -- Charge_type and minimum_cell_voltage have a good correlation. that's pretty much how charge_type seems defined.
     * -- Same as above for minimum_battery temperature
  #### Interesting:
      * -- Charge type "0F" has a much bigger range of temperatures than the other 3 charge_type(s)
      * -- Charge type "0" is concentrated almost all in the longitude range around -100.. (Hmm.. perhaps local power supply restrictions??)
      * -- Charge type "3" has the most variation in "charging at the end of remaining capacity". perhaps that's the oldest cars and the batteries have weakened enough to have varied capacity??
      
  

```python
sns.pairplot(df, hue='charge_type')
```

```python
df.charge_type.unique()
```

```python
interesting_cols = ['charging_history__charging_at_the_start_level_indicator',
       'charging_history__charging_at_the_end_of_level_indicator',
       'charging_history__charging_at_the_start_of_the_remaining_capacity',
       'charging_history__charging_at_the_end_of_the_remaining_capacity',
       'charging_history__power_usage', 'charge_type',
       'rapidly_charging_history__charging_position_information___latitude',
       'charging_history__charging_position_information___longitude',
       'charging_history__start_charging_when_the_battery_average_temperature',
       'charging_history___start_charging_when_the_battery_maximum_temperature',
       'charging_history__start_charging_when_the_battery_minimum_temperature',
       'charging_history__end-of-charge_when_the_battery_average_temperature',
       'charging_history__end-of-charge_when_the_battery_maximum_temperature',
       'charging_history__end-of-charge_when_the_battery_minimum_temperature',
       'Charging_history__Start_charging_when_the_battery_average_cell_voltage',
       'charging_history__Start_charging_when_the_battery_maximum_cell_voltage',
       'charging_history__Start_charging_when_the_battery_minimum_cell_voltage',
       'charging_history__Charging_at_the_end_of_the_current_integration',
       'charging_in_the_charging_history__IGN_minus_OFF_number',
       'charging_history__charging_end_time_IR_sensor_value',
       'charging_history__accessory_information_signal'
                   ]

with sns.axes_style('white'):
    sns.jointplot(df['charging_history___start_charging_when_the_battery_maximum_temperature'], 
                  df["charging_history__power_usage"],  kind='scatter')
    plt.show()

for each in [('charging_history__start_charging_when_the_battery_average_temperature','charging_history__end-of-charge_when_the_battery_average_temperature'), 
             ('charging_history__start_charging_when_the_battery_average_temperature','Charging_history__Start_charging_when_the_battery_average_cell_voltage'),
             ('charging_history__Start_charging_when_the_battery_maximum_cell_voltage','charging_history__end-of-charge_when_the_battery_maximum_temperature'),
             ('charging_history__charging_at_the_end_of_the_remaining_capacity', 'charging_history__charging_at_the_end_of_level_indicator'),
             ('charging_history__charging_at_the_start_level_indicator','charging_history__Charging_at_the_end_of_the_current_integration'), 
            ]:
    sns.jointplot(each[0], each[1], data=df, kind='scatter')
    plt.show()
```

```python
from datetime import datetime
import pytz
def create_start_date_obj(x):
    print(x[4],x[5])
    return datetime(year=int(x[3]), month=int(x[4]),
                    day=int(x[5]), hour=int(x[6]),
                    minute=int(x[7]), tzinfo=pytz.utc)

def create_term_date_obj(x):
    print(x)
    return datetime(year=int(x[10]), month=int(x[11]),
                    day=int(x[12]), hour=int(x[13]),
                    minute=int(x[14]), tzinfo=pytz.utc)


df['charging_start_time'] = df.apply(lambda x: create_start_date_obj(x), axis=1)
df['charging_end_time'] = df.apply(lambda x: create_term_date_obj(x), axis=1)
mod_df.charging_history_charge_start_time_day = mod_df.loc[mod_df.charging_history_charge_start_time_month==2,
        'charging_history_charge_start_time_day'].replace(to_replace=[29, 30,31], value=[28, 28, 28])

mod_df.charging_history_charge_term_time_day = mod_df.loc[mod_df.charging_history_charge_term_time_month==2,
        'charging_history_charge_term_time_day'].replace(to_replace=[29, 30, 31], value=[28,28,28])
```

```python

```
