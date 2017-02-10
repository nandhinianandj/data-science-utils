# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import output_notebook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from datascienceutils import analyze
from datascienceutils import predictiveModels as pm
from datascienceutils import sklearnUtils as sku
from datascienceutils import settings
# In[2]:

df = pd.read_csv('../data/train_63qYitG.csv')
test_df = pd.read_csv('../data/test_XaoFywY.csv')
test_IDS = test_df.Trip_ID
test_df.drop('Trip_ID',1, inplace=True)

gle = LabelEncoder()
gle.fit(df['Gender'])
df['Gender'] = gle.transform(df['Gender'])
test_df['Gender'] = gle.transform(test_df['Gender'])

lse = LabelEncoder()
df['Confidence_Life_Style_Index'].fillna('B', inplace=True)
test_df['Confidence_Life_Style_Index'].fillna('B', inplace=True)
lse.fit(df['Confidence_Life_Style_Index'])
df['Confidence_Life_Style_Index'] = lse.transform(df['Confidence_Life_Style_Index'])
test_df['Confidence_Life_Style_Index'] = lse.transform(test_df['Confidence_Life_Style_Index'])


dte = LabelEncoder()
dte.fit(df['Destination_Type'])
df['Destination_Type'] = dte.transform(df['Destination_Type'])
test_df['Destination_Type'] = dte.transform(test_df['Destination_Type'])


toce = LabelEncoder()
df['Type_of_Cab'].fillna('B', inplace=True)
test_df['Type_of_Cab'].fillna('B', inplace=True)
toce.fit(df['Type_of_Cab'])
df['Type_of_Cab'] = toce.transform(df['Type_of_Cab'])
test_df['Type_of_Cab'] = toce.transform(test_df['Type_of_Cab'])

target = df.Surge_Pricing_Type
df.drop(['Trip_ID', 'Surge_Pricing_Type'], 1, inplace=True)

df.fillna(df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)
#X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.33)
X_train = df
y_train = target

X_test = test_df
# In[3]:

settings.MODELS_BASE_PATH='../models'
# Train the model using the training sets
#import pdb; pdb.set_trace()
#lin_model = pm.train(X_train, y_train, 'LinearRegression')
#sku.dump_model(lin_model, 'lin_reg', model_params={'model_type':'linReg'})
#print('Coefficients: \n', lin_model.coef_)
## The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((lin_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % lin_model.score(X_test, y_test))


# In[ ]:

## Train the model using the training sets
#log_model = pm.train(X_train, y_train, 'logisticRegression')
#sku.dump_model(lin_model, 'log_reg', model_params={'model_type':'logReg'})

#print('Coefficients: \n', log_model.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((log_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % log_model.score(X_test, y_test))


# In[ ]:

print("Training Random Forest Model")
# Train the model using the training sets
rf_model = pm.train(X_train, y_train, 'randomForest')

sku.dump_model(rf_model, 'random_f', model_params={'model_type':'randomForest'})

predictions = rf_model.predict(test_df)
new_df = pd.DataFrame()
new_df['Trip_ID'] = test_IDS
new_df['Surge_Pricing_Type'] = predictions
new_df.to_csv('./random_forest_predictions.csv')
#print('Coefficients: \n', rf_model.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((rf_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % rf_model.score(X_test, y_test))


# In[ ]:

# Train the model using the training sets
sgd_model = pm.train(X_train, y_train, 'sgd')
print("Training SGD Model")
sgd_model.fit(X_train, y_train)
sku.dump_model(sgd_model, 'sgd', model_params={'model_type':'sgd'})

predictions = sgd_model.predict(test_df)
new_df = pd.DataFrame()
new_df['Trip_ID'] = test_IDS
new_df['Surge_Pricing_Type'] = predictions
new_df.to_csv('./sgd_predictions.csv')
#print('Coefficients: \n', rf_model.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((sgd_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % sgd_model.score(X_test, y_test))


# In[ ]:

# Train the model using the training sets
print("Training XGBOOST Model")
xgb_model = pm.train(X_train, y_train, 'xgboost')
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(test_df)
new_df = pd.DataFrame()
new_df['Trip_ID'] = test_IDS
new_df['Surge_Pricing_Type'] = predictions
new_df.to_csv('./xgboost_predictions.csv')
#print('Coefficients: \n', rf_model.coef_)
sku.dump_model(xgb_model, 'xgboost', model_params={'model_type':'xgboost'})
# The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((xgb_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % xgb_model.score(X_test, y_test))


print("Training knn Model")
# Train the model using the training sets
knn_model = pm.train(X_train, y_train, 'knn')
knn_model.fit(X_train, y_train)
predictions = knn_model.predict(test_df)
new_df = pd.DataFrame()
new_df['Trip_ID'] = test_IDS
new_df['Surge_Pricing_Type'] = predictions
new_df.to_csv('./knn_predictions.csv')
#print('Coefficients: \n', rf_model.coef_)
sku.dump_model(knn_model, 'knn', model_params={'model_type':'knn'})
# The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((knn_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % knn_model.score(X_test, y_test))


# Train the SVM model using the training sets
#print("Training SVM Model")
#svm_model = pm.train(X_train, y_train, 'svm')
#svm_model.fit(X_train, y_train)
#predictions = svm_model.predict(test_df)
#new_df = pd.DataFrame()
#new_df['Trip_ID'] = test_IDS
#new_df['Surge_Pricing_Type'] = predictions
#new_df.to_csv('./svm_predictions.csv')
#
##print('Coefficients: \n', rf_model.coef_)
#sku.dump_model(svm_model, 'svm', model_params={'model_type':'svm'})
## The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((svm_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % svm_model.score(X_test, y_test))

#print("Training multinomialNB Model")
## Train the model using the training sets
#mnb_model = pm.train(X_train, y_train, 'multinomialNB')
#
#sku.dump_model(mnb_model, 'mnb', model_params={'model_type':'mnb'})
#print('Coefficients: \n', mnb_model.coef_)
## The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((mnb_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % mnb_model.score(X_test, y_test))
#
#
## Train the model using the training sets
#print("Training bernoulliNB Model")
#bnb_model = pm.train(X_train, y_train, 'bernoulliNB')
#bnb_model.fit(X_train, y_train)
#sku.dump_model(lin_model, 'bnb', model_params={'model_type':'bnb'})
## The mean squared error
#print("Mean squared error: %.2f"
#      % np.mean((bnb_model.predict(X_test) - y_test) ** 2))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % bnb_model.score(X_test, y_test))



# ## Linear Regression is top with MSE: 2548.07
# ## But we know this is a linear regression data set in the first place
#
# ## Of the non-linear  models
# ## Clearly xgboost takes the cake with MSE: 4906 runs in 5.94s
# ## Followed by knn MSE: 5640.65
#
#
# ## I heard about [lightgbm](https://github.com/ArdalanM/pyLightGBM)  and wanted to try it.
# ## So check it out MSE: 5066.17 and runs in 194ms
#
# ## Wow that's multiple orders of magnitude faster and only about 10% more error.. May be lightgbm will work very well for linear patterns. Need to check for other patterns and if it keeps similar trade-offs, then it'll change the market
