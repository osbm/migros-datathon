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
    print(df.shape)        
    # add customer data according to individualnumber
    df = df.merge(customer_df, on="individualnumber", how="left")

    df = df.merge(customer_account_df, on="individualnumber", how="left")
    # print all the new rows that are added
    
    # some customers have multiple cardnumbers
    # we will use the first cardnumber
    
    df = df.drop_duplicates(subset="individualnumber")

    
    print(df.shape)        
    
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


def pipeline(df):
    df = df.copy()
    #print(df.shape)
    df = add_customer_data(df)
    #print(df.shape)
    df = add_number_of_transactions(df)
    #print(df.shape)
    df = add_total_amount_spent(df)
    #print(df.shape)
    return df


train_df = pipeline(train_df)
train_df.to_csv("preprocessed_train.csv", index=False)

test_df = pipeline(test_df)
test_df.to_csv("preprocessed_test.csv", index=False)