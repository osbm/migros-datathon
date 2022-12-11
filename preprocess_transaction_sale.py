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
product_group_df = pd.read_csv(dataset_folder / "product_groups.csv")
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

transaction_sale_df["total_expenditure"] = transaction_sale_df["amount"] * transaction_sale_df["quantity"]

# adding category number to transaction_sale_df
# add category nummber according to category_level_1. category_level_2, category_level_3, and category_level_4 
for i, row in product_group_df.iterrows():
    level_1_filter = transaction_sale_df.category_level_1 == row.category_level_1
    level_2_filter = transaction_sale_df.category_level_2 == row.category_level_2
    level_3_filter = transaction_sale_df.category_level_3 == row.category_level_3
    level_4_filter = transaction_sale_df.category_level_4 == row.category_level_4
    transaction_sale_df.loc[level_1_filter & level_2_filter & level_3_filter & level_4_filter, "category_number"] = row.category_number.astype(int)

# add general category according to category number

print(transaction_sale_df.shape)
transaction_sale_df = transaction_sale_df.merge(category_lookup_df, on="category_number", how="left")

# merge the two dataframes
# transaction_sale_df = pd.merge(
#     transaction_sale_df,
#     product_group_df,
#     on=["category_level_1", "category_level_2", "category_level_3", "category_level_4"],
#     how="left"
# )
# group by category number and calculate the mean for each group
#transaction_sale_df = transaction_sale_df.groupby("category_number").mean()
print(transaction_sale_df.shape)


# save the transaction_sale_df
transaction_sale_df.to_csv("transaction_sale_preprocessed.csv", index=False)
