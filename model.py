import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

if not os.path.exists("preprocessed_train.csv"):
    raise Exception("preprocessed_train.csv does not exist. Please run preprocessing.py first")

train_df = pd.read_csv("preprocessed_train.csv")

y = train_df["response"]
X = train_df.drop("response", axis=1)

estimators = [
    #("logreg", LogisticRegression()),
    #("rf", RandomForestClassifier()),
    ("adaboost", AdaBoostClassifier()),
    ("xgb", XGBClassifier()),
    ("lgbm", LGBMClassifier()),
    ("cat", CatBoostClassifier(verbose=0, allow_writing_files=False)),
]

ensemble = VotingClassifier(estimators)

model = Pipeline(
    [
        ("over_sampling", SMOTE(random_state=42)),
        ("ensemble", ensemble),
    ]
)

# define the grid search parameters
param_grid = {
    "ensemble__adaboost__n_estimators": [50, 100, 200],
    "ensemble__xgb__n_estimators": [50, 100, 200],
    "ensemble__lgbm__n_estimators": [50, 100, 200],
    "ensemble__cat__n_estimators": [50, 100, 200],
}
""" param_grid = {
    "adaboost__n_estimators": [50, 100, 200],
    "xgb__n_estimators": [50, 100, 200],
    "lgbm__n_estimators": [50, 100, 200],
    "cat__n_estimators": [50, 100, 200],
}
 """
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="f1", verbose=1)
grid_result = grid.fit(X, y)

print("Best score:", grid_result.best_score_)
print("Best params:", grid_result.best_params_)

test_df = pd.read_csv("preprocessed_test.csv")
y_pred = grid_result.predict(test_df)

test_df["response"] = y_pred # add response
test_df = test_df[["response"]] # drop every other column
test_df["individualnumber"] = pd.read_csv("data/test.csv")["individualnumber"] # add individualnumber
test_df.to_csv("submission.csv", index=False)
