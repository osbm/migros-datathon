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
    X, y, test_size=0.1, random_state=42, stratify=y
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
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print("ensemble accuracy: ", accuracy_score(y_test, y_pred))
print("ensemble f1 score: ", f1_score(y_test, y_pred))

test_df = pd.read_csv("preprocessed_test.csv")
lgbm_pred = ensemble.predict(test_df)

test_df["response"] = lgbm_pred # add response
test_df = test_df[["response"]] # drop every other column
test_df["individualnumber"] = pd.read_csv("data/test.csv")["individualnumber"] # add individualnumber
test_df.to_csv("submission.csv", index=False)
