import pandas as pd
import numpy as np
train_df = pd.read_csv("preprocessed_train.csv")
test_df = pd.read_csv("preprocessed_test.csv")

# model time and use f1 score
# logreg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X = train_df.drop(["response", "individualnumber", "cardnumber"], axis=1)
y = train_df.response

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# categorical features
category_cols = [
    "gender",
    "city_code",
]

# one hot encode categorical features
X_train = pd.get_dummies(X_train, columns=category_cols)
X_test = pd.get_dummies(X_test, columns=category_cols)

# fill missing values with mean
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# find missing columns in test set
missing_cols = set(X_train.columns) - set(X_test.columns)

# add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0

# ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("logreg accuracy: ", accuracy_score(y_test, y_pred))
print("logreg f1 score: ", f1_score(y_test, y_pred))

# random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("rf accuracy: ", accuracy_score(y_test, y_pred))
print("rf f1 score: ", f1_score(y_test, y_pred))

# xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("xgb accuracy: ", accuracy_score(y_test, y_pred))
print("xgb f1 score: ", f1_score(y_test, y_pred))

# lightgbm
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print("lgbm accuracy: ", accuracy_score(y_test, y_pred))
print("lgbm f1 score: ", f1_score(y_test, y_pred))

lgbm_pred = y_pred

# catboost
from catboost import CatBoostClassifier

cat = CatBoostClassifier()
cat.fit(X_train, y_train)
y_pred = cat.predict(X_test)
print("cat accuracy: ", accuracy_score(y_test, y_pred))
print("cat f1 score: ", f1_score(y_test, y_pred))


# import ensemble classifier
from sklearn.ensemble import VotingClassifier

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(("logreg", model1))
model2 = RandomForestClassifier()
estimators.append(("rf", model2))
model3 = XGBClassifier()
estimators.append(("xgb", model3))
model4 = LGBMClassifier()
estimators.append(("lgbm", model4))
model5 = CatBoostClassifier()
estimators.append(("cat", model5))

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

# add a missing column in test set with default value equal to 0
for c in missing_cols:
    X[c] = 0

# ensure the order of column in the test set is in the same order than in train set
X = X[X_train.columns]

lgbm_pred = ensemble.predict(X)

test_df["response"] = lgbm_pred
test_df = test_df[["individualnumber", "response"]]
test_df.to_csv("submission.csv", index=False)

#lgbm_pred