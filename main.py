import pandas as pd
import numpy as np


from pathlib import Path

dataset_folder = Path('data/')

# Load the dataframes
train_df = pd.read_csv(dataset_folder / 'train.csv')
test_df = pd.read_csv(dataset_folder / 'test.csv')

# auxiliary dataframes 
customer_df = pd.read_csv(dataset_folder / 'customer.csv')
customer_account_df = pd.read_csv(dataset_folder / 'customer_account.csv')
category_lookup_df = pd.read_csv(dataset_folder / 'genel_categories.csv')

def add_customer_data(df):
    df = df.merge(customer_df, on='individualnumber', how='left')
    df = df.merge(customer_account_df, on='individualnumber', how='left')
    return df




