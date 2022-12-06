import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import VotingClassifier

train_df = pd.read_csv("preprocessed_train.csv")
test_df = pd.read_csv("preprocessed_test.csv")

# model time and use f1 score

# drop response variable from train set
y = train_df["response"]
X = train_df.drop("response", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("logreg accuracy: ", accuracy_score(y_test, y_pred))
print("logreg f1 score: ", f1_score(y_test, y_pred))

# random forest


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("rf accuracy: ", accuracy_score(y_test, y_pred))
print("rf f1 score: ", f1_score(y_test, y_pred))

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("xgb accuracy: ", accuracy_score(y_test, y_pred))
print("xgb f1 score: ", f1_score(y_test, y_pred))

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print("lgbm accuracy: ", accuracy_score(y_test, y_pred))
print("lgbm f1 score: ", f1_score(y_test, y_pred))

lgbm_pred = y_pred


cat = CatBoostClassifier()
cat.fit(X_train, y_train)
y_pred = cat.predict(X_test)
print("cat accuracy: ", accuracy_score(y_test, y_pred))
print("cat f1 score: ", f1_score(y_test, y_pred))


# create the sub models
estimators = [
    ("logreg", LogisticRegression()),
    ("rf", RandomForestClassifier()),
    ("xgb", XGBClassifier()),
    ("lgbm", LGBMClassifier()),
    ("cat", CatBoostClassifier()),
]

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
print("ensemble accuracy: ", accuracy_score(y_test, y_pred))
print("ensemble f1 score: ", f1_score(y_test, y_pred))


X = test_df.drop(["individualnumber", "cardnumber"], axis=1)

# categorical features
category_cols = [
    "gender",
    "city_code",
]

# one hot encode categorical features
X = pd.get_dummies(X, columns=category_cols)

# fill missing values with mean
X = X.fillna(X.mean())


# find missing columns in test set
missing_cols = set(X_train.columns) - set(X.columns)
print("missing", missing_cols)
# add a missing column in test set with default value equal to 0
for c in missing_cols:
    X[c] = 0

lgbm_pred = ensemble.predict(X)
test_df["response"] = lgbm_pred
test_df = test_df[["individualnumber", "response"]]
test_df.to_csv("submission.csv", index=False)
