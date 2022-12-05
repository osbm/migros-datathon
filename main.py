import pandas as pd
import numpy as np

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
    df = df.merge(customer_df, on="individualnumber", how="left")
    df = df.merge(customer_account_df, on="individualnumber", how="left")

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


def pipeline(df):
    df = df.copy()
    df = add_customer_data(df)
    df = add_number_of_transactions(df)
    df = add_total_amount_spent(df)
    return df


train_df = pipeline(train_df)
train_df.to_csv("preprocessed_train.csv", index=False)
