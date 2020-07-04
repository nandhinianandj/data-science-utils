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
from sklearn import datasets
import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import output_notebook

from dsu import analyze
from dsu import explain
from dsu import predictiveModels as pm

output_notebook()
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
dir(diabetes)
```

```python
df = pd.DataFrame(diabetes.data, columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
target = diabetes.target
analyze.correlation_analyze(df, 'age', 'bmi')
```

```python

# Train the model using the training sets
lin_model = pm.train(diabetes_X_train, diabetes_y_train, 'LinearRegression')

print('Coefficients: \n', lin_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lin_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lin_model.score(diabetes_X_test, diabetes_y_test))

#explain.interpret(df, lin_model)
```

```python
# Train the model using the training sets
log_model = pm.train(diabetes_X_train, diabetes_y_train, 'LogisticRegression')

#print('Coefficients: \n', log_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((log_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % log_model.score(diabetes_X_test, diabetes_y_test))
```

```python
from dsu import sklearnUtils as sku
from dsu import settings
settings.MODELS_BASE_PATH='../models'
params={'model_type': 'logarithmicRegression', 
        'output_type':'binary', 
        'input_metadata': 'testing'}
sku.dump_model(log_model, 'logarithimic_regression',params)


```

```python
params={'model_type': 'linearRegression', 
        'output_type':'float', 
        'input_metadata': 'testing'}
sku.dump_model(log_model, 'linear_regression',params)
```

```python
# Train the model using the training sets
log_model = pm.train(diabetes_X_train, diabetes_y_train, 'IsotonicRegression')

#print('Coefficients: \n', log_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((log_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % log_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
log_model = pm.train(diabetes_X_train, diabetes_y_train, 'SVMRegression')

#print('Coefficients: \n', log_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((log_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % log_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
log_model = pm.train(diabetes_X_train, diabetes_y_train, 'RANSACRegression')

#print('Coefficients: \n', log_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((log_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % log_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
rf_model = pm.train(diabetes_X_train, diabetes_y_train, 'randomForest')

#print('Coefficients: \n', rf_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((rf_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rf_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
sgd_model = pm.train(diabetes_X_train, diabetes_y_train, 'sgd')
sgd_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((sgd_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % sgd_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
xgb_model = pm.train(diabetes_X_train, diabetes_y_train, 'xgboost')
xgb_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((xgb_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % xgb_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
svm_model = pm.train(diabetes_X_train, diabetes_y_train, 'svm')
svm_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((svm_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % svm_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
bnb_model = pm.train(diabetes_X_train, diabetes_y_train, 'bernoulliNB')
bnb_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((bnb_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % bnb_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
knn_model = pm.train(diabetes_X_train, diabetes_y_train, 'knn')
knn_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((knn_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % knn_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
kde_model = pm.train(diabetes_X_train, diabetes_y_train, 'kde')
kde_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((kde_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % kde_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
kde_model = pm.train(diabetes_X_train, diabetes_y_train, 'kde')
kde_model.fit(diabetes_X_train, diabetes_y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((kde_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % kde_model.score(diabetes_X_test, diabetes_y_test))
```

```python
# Train the model using the training sets
mnb_model = pm.train(diabetes_X_train, diabetes_y_train, 'multinomialNB')

print('Coefficients: \n', mnb_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((mnb_model.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mnb_model.score(diabetes_X_test, diabetes_y_test))
```

```python
X, y = datasets.load_diabetes(return_X_y=True)
X.shape
```

<!-- #region -->
## Linear Regression is top with MSE: 2548.07
## But we know this is a linear regression data set in the first place

## Of the non-linear  models
## Clearly xgboost takes the cake with MSE: 4906 runs in 5.94s
## Followed by knn MSE: 5640.65


## I heard about [lightgbm](https://github.com/ArdalanM/pyLightGBM)  and wanted to try it. 
## So check it out MSE: 5066.17 and runs in 194ms

## Wow that's multiple orders of magnitude faster and only about 10% more error.. May be lightgbm will work very well for linear patterns. Need to check for other patterns and if it keeps similar trade-offs, then it'll change the market
<!-- #endregion -->
