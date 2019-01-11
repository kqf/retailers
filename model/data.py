from itertools import product

import numpy as np
import pandas as pd


def outer_product(clients=2000, products=40, days=50):
    return product(
        np.arange(clients),
        np.arange(products),
        np.arange(days)
    )


def extend_dataset(df):
    dummy = pd.DataFrame(list(outer_product()),
                         columns=["i", "j", "t"])

    full_set = pd.merge(df, dummy, how="outer",
                        left_on=["i", "j", "t"],
                        right_on=["i", "j", "t"])

    # It seems like the product was advertised for every i
    time_product = full_set.groupby(["j", "t"], as_index=False)
    full_set["advertised"] = time_product["advertised"].transform(
        lambda x: x.mean())

    # Handle unseen combinations
    product = full_set.groupby(["j"], as_index=False)
    full_set["advertised"] = product["advertised"].transform(
        lambda x: x.fillna(x.median())
    )

    # Price seems to be the same
    full_set["price"] = time_product["price"].transform(
        lambda x: x.mean())

    full_set["price"] = product["price"].transform(
        lambda x: x.fillna(x.median())
    )

    # Create a target variable
    full_set["purchase"] = full_set["purchase"].fillna(False)
    return full_set


def downsample(df):
    # return df
    print("Before downlsampling", df.size)
    original = df[df.purchase == 1]
    sampled = df[df.purchase == 0].sample(n=len(original))
    concatenated = pd.concat([original, sampled])
    print("After downlsampling", concatenated.size)
    return concatenated


def prepare_dataset(path):
    df = pd.read_csv(path / "train.csv")

    # Create target variable
    df["purchase"] = True

    # Add empty bins
    return extend_dataset(df)
