import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path

dataset_folder = Path("data/")

# Load the dataframes
train_df = pd.read_csv(dataset_folder / "train.csv")
test_df = pd.read_csv(dataset_folder / "test.csv")

# auxiliary dataframes
customer_df = pd.read_csv(dataset_folder / "customer.csv")
customer_account_df = pd.read_csv(dataset_folder / "customeraccount.csv")
category_lookup_df = pd.read_csv(dataset_folder / "genel_kategoriler.csv")
transaction_header_df = pd.read_csv(dataset_folder / "transaction_header.csv")
transaction_sale_df = pd.read_csv(
    dataset_folder / "transaction_sale" / "transaction_sale.csv"
)
transaction_sale_preprocessed_df = pd.read_csv("transaction_sale_preprocessed.csv")


def add_customer_data(df):
    df = df.copy()
    # add customer data according to individualnumber
    df = df.merge(customer_df, on="individualnumber", how="left")

    df = df.merge(customer_account_df, on="individualnumber", how="left")
    # print all the new rows that are added

    # some customers have multiple cardnumbers
    # we will use the first cardnumber

    df = df.drop_duplicates(subset="individualnumber")

    # this operation changes row number be
    return df


def add_number_of_transactions(df):
    df = df.copy()
    counts = transaction_header_df.cardnumber.value_counts()
    df["number_of_transactions"] = df.cardnumber.map(counts)
    return df


def add_total_amount_spent(df):
    df = df.copy()
    df["total_amount_spent"] = df.cardnumber.map(
        transaction_sale_preprocessed_df.groupby("cardnumber").amount.sum()
    )
    return df


def one_hot_encode(df, categorical_columns):
    df = df.copy()
    df = pd.get_dummies(df, columns=categorical_columns)
    return df


def resample(df):
    # this function resamples the data to have equal number of 0 and 1
    df = df.copy()
    df_0 = df[df.response == 0]
    df_1 = df[df.response == 1]
    df_0 = df_0.sample(n=df_1.shape[0], random_state=42)
    df = pd.concat([df_0, df_1], axis=0)
    return df


def fill_na(df):
    df = df.copy()
    df["total_amount_spent"] = df["total_amount_spent"].fillna(0)
    df["number_of_transactions"] = df["number_of_transactions"].fillna(0)
    df = df.fillna(df.mean())
    return df


def add_missing_columns(df):
    df = df.copy()
    missing_cols = set(train_df.columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    # reorder the columns
    df = df[train_df.columns]
    return df


def apply_SMOTE(df):

    df = df.copy()
    X = df.drop("response", axis=1)
    y = df.response
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    df = pd.concat([X_res, y_res], axis=1)
    return df


def drop_columns(df, columns=["cardnumber", "individualnumber"]):
    df = df.copy()
    df = df.drop(columns, axis=1)
    return df


def pipeline(df, train=True):
    category_cols = [
        "gender",
        "city_code",
    ]
    df = df.copy()
    df = add_customer_data(df)
    df = add_number_of_transactions(df)
    df = add_total_amount_spent(df)
    df = drop_columns(df)
    df = one_hot_encode(df, category_cols)
    df = fill_na(df)
    df = add_missing_columns(df)

    if train:
        # df = resample(df)
        df = apply_SMOTE(df)
    return df


train_df = pipeline(train_df, train=True)
train_df.to_csv("preprocessed_train.csv", index=False)

test_df = pipeline(test_df, train=False)
test_df.to_csv("preprocessed_test.csv", index=False)
