import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os

from imblearn.over_sampling import SMOTE

if not os.path.exists("preprocessed_train.csv"):
    raise Exception("preprocessed_train.csv does not exist. Please run preprocessing.py first")

train_df = pd.read_csv("preprocessed_train.csv")

# drop response variable from train set
y = train_df["response"]
X = train_df.drop("response", axis=1)

# use stratified sampling to split the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

# create the sub models
estimators = [
    #("logreg", LogisticRegression()),
    #("rf", RandomForestClassifier()),
    ("adaboost", AdaBoostClassifier()),
    ("xgb", XGBClassifier()),
    ("lgbm", LGBMClassifier()),
    ("cat", CatBoostClassifier(verbose=0, allow_writing_files=False)),
]

# create the ensemble model
ensemble = VotingClassifier(estimators)

from sklearn.model_selection import cross_val_score
# grid search
from sklearn.model_selection import GridSearchCV

# define the grid search parameters
param_grid = {
    "adaboost__n_estimators": [50, 100, 200],
    "xgb__n_estimators": [50, 100, 200],
    "lgbm__n_estimators": [50, 100, 200],
    "cat__n_estimators": [50, 100, 200],
}

# grid search
grid = GridSearchCV(estimator=ensemble, param_grid=param_grid, cv=5, scoring="f1", verbose=1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# predict
y_pred = grid_result.predict(X_test)





test_df = pd.read_csv("preprocessed_test.csv")
#lgbm_pred = ensemble.predict(test_df)

test_df["response"] = y_pred # add response
test_df = test_df[["response"]] # drop every other column
test_df["individualnumber"] = pd.read_csv("data/test.csv")["individualnumber"] # add individualnumber
test_df.to_csv("submission.csv", index=False)
