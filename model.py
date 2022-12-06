# auto model using pycaret

from pycaret.classification import *
import pandas as pd
import numpy as np

from pathlib import Path

train_df = pd.read_csv("preprocessed_train.csv")
test_df = pd.read_csv("preprocessed_test.csv")








# drop individualnumber and cardnumber
#train_df = train_df.drop(["individualnumber", "cardnumber"], axis=1)
#test_df = test_df.drop(["individualnumber", "cardnumber"], axis=1)



category_cols = [
    "category_number",
    "gender",
]

exclude_cols = ["individualnumber", "cardnumber"]

experiment = setup(
    data=train_df,
    target="response",
    categorical_features=category_cols,
)

best_model = compare_models()

# save model
save_model(best_model, "best_model")

# predict
predictions = predict_model(best_model, data=test_df)


predictions = predictions[["individualnumber", "Label"]]
predictions = predictions.rename(columns={"Label": "response"})
predictions.to_csv("submission.csv", index=False)
