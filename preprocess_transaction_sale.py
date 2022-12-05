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

transaction_sale_df["cardnumber"] = transaction_sale_df.basketid.map(
    transaction_header_df.set_index("basketid").cardnumber
)

# add individualnumber to transaction_sale_df
transaction_sale_df["individualnumber"] = transaction_sale_df.cardnumber.map(
    customer_account_df.set_index("cardnumber").individualnumber
)

# save the transaction_sale_df
transaction_sale_df.to_csv("transaction_sale_preprocessed.csv", index=False)
