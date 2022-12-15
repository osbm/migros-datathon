import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#from fastai.tabular.transform import (add_datepart,)


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
transaction_sale_preprocessed_df = pd.read_csv(
    "transaction_sale_preprocessed.csv")


def add_day_of_month(df):
    df = df.copy()
    transactions_head_df = transaction_header_df.copy()
    day = transactions_head_df["date_of_transaction"].str.split(
        "-", expand=True)
    q = transactions_head_df[transactions_head_df.cardnumber.isin(
        [i for i in transaction_header_df["cardnumber"]])]
    print()
    df["day"] = (day[2])
    return df


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
    # there are multiple rows for each individualnumber
    # we will sum the total expenditure for each individualnumber
    print(df.shape)
    tsp_df = transaction_sale_preprocessed_df.groupby(
        "individualnumber").agg({"total_expenditure": "sum"})
    tsp_df.reset_index(inplace=True)

    # sum
    df = df.merge(tsp_df, on="individualnumber", how="left")
    print(df.shape)

    return df


def add_number_of_cards(df):
    df = df.copy()
    counts = customer_account_df.individualnumber.value_counts()
    df["number_of_cards"] = df.individualnumber.map(counts)
    return df


def add_general_category(df):
    df = df.copy()
    df = df.merge(category_lookup_df, on="category_number", how="left")

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
    df["total_expenditure"] = df["total_expenditure"].fillna(0)
    df["number_of_transactions"] = df["number_of_transactions"].fillna(0)
    df = df.fillna(df.mean())
    return df


def add_missing_columns(df):
    df = df.copy()

    train_df = pd.read_csv("preprocessed_train.csv")

    missing_cols = set(train_df.columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    # reorder the columns
    df = df[train_df.columns]
    # drop response column
    df = df.drop("response", axis=1)
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


def add_most_common_cities(df):
    df = df.copy()
    df["is_istanbul"] = df.city_code.apply(lambda x: 1 if x == 34 else 0)
    df["is_ankara"] = df.city_code.apply(lambda x: 1 if x == 6 else 0)
    df["is_izmir"] = df.city_code.apply(lambda x: 1 if x == 35 else 0)
    df["is_antalya"] = df.city_code.apply(lambda x: 1 if x == 7 else 0)
    # if the cift code outside 1-81, then it is unknown
    df["is_unk_city"] = df.city_code.apply(
        lambda x: 1 if x not in range(1, 82) else 0)

    # drop city_code column
    df = df.drop("city_code", axis=1)
    return df


def add_birth_decade(df):
    df = df.copy()
    df["is_90s"] = df.birthdate.apply(
        lambda x: 1 if x >= 1990 and x < 2000 else 0)
    df["is_80s"] = df.birthdate.apply(
        lambda x: 1 if x >= 1980 and x < 1990 else 0)
    df["is_70s"] = df.birthdate.apply(
        lambda x: 1 if x >= 1970 and x < 1980 else 0)
    df["is_2000s"] = df.birthdate.apply(
        lambda x: 1 if x >= 2000 and x < 2010 else 0)
    return df


def add_month_quarter(df):
    df = df.copy()


def pipeline(df, train=True):
    category_cols = [
        "gender",
        "genel_kategori",
        # "city_code", # generates ~80 columns after one-hot encoding
    ]

    df = df.copy()
    df = add_day_of_month(df)
    df = add_customer_data(df)
    df = add_number_of_transactions(df)
    df = add_total_amount_spent(df)
    df = add_number_of_cards(df)
    df = add_general_category(df)
    df = add_most_common_cities(df)
    df = drop_columns(df)
    df = one_hot_encode(df, category_cols)
    df = fill_na(df)

    if train:
        # we are splitting the data later, so we shouldn't resample before splitting
        # because otherwise we will have data leakage
        # df = resample(df)
        #df = apply_SMOTE(df)
        pass
    else:
        df = add_missing_columns(df)

    return df


if __name__ == "__main__":

    train_df = pipeline(train_df, train=True)
    # show correlation matrix according to response using seaborn

    sns.heatmap(train_df.corr()[["response"]].sort_values(
        by="response"), annot=True)
    plt.savefig("correlation_matrix.png")

    train_df.to_csv("preprocessed_train.csv", index=False)

    test_df = pipeline(test_df, train=False)
    test_df.to_csv("preprocessed_test.csv", index=False)
    print(train_df.shape)
    print(test_df.shape)
