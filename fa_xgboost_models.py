import json
import bokeh
import numpy as np
import pandas as pd
import random

from bokeh.plotting import output_notebook
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from datascienceutils import analyze
from datascienceutils import predictiveModels as pm
from datascienceutils import sklearnUtils as sku
from datascienceutils import settings
# In[2]:

with open('../data/training.json', 'r') as fd:
    inp = fd.readlines()
rows = list()
for idx, each in enumerate(inp):
    rows.append(json.loads(each))
df = pd.DataFrame(rows)


###########################################################
# DATA CLEANUP
###########################################################
# Picking randomly from one of top few modes, and for too many modes just replace with -1
df['Mathematics'].fillna(random.choice([1, 2, 3, 8]))
df['Biology'].fillna(-1, inplace=True)
df['Physics'].fillna(-1, inplace=True)
df['PhysicalEducation'].fillna(-1, inplace=True)
df['English'].fillna(random.choice([1, 2, 3, 4]))
df['Economics'].fillna(random.choice([1.0, 2.0, 3.0, 4.0]))
df['ComputerScience'].fillna(-1, inplace=True)
df['BusinessStudies'].fillna(random.choice([1.0, 2.0, 3.0, 4.0, 5.0]))
df['Chemistry'].fillna(random.choice([1.0, 2.0, 3.0, 4.0, 5.0]))

target = df['Mathematics']
df.drop(['Mathematics','serial'], 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.33)


settings.MODELS_BASE_PATH='../models'

print("Training XGBOOST Model")
parameters= {
'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
 }
import xgboost
xgb_model = xgboost.XGBClassifier()
#xgb_model = pm.train(X_train, y_train, 'xgboost')
xgb_model = GridSearchCV(xgb_model, parameters)
xgb_model.fit(X_train, y_train)

predictions = xgb_model.predict(test_df)
from sklearn.metrics import r2_score
print(r2_score(y_test, predictions))
new_df = pd.DataFrame()
#new_df.to_csv('./xgboost_predictions.csv')
#sku.dump_model(mod, 'xgboost', model_params={'model_type':'xgboost_gs_cv'})
