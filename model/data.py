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


def downsample(df, skip=False):
    if skip:
        return df
    print("Before downlsampling", len(df))
    original = df[df.purchase == 1]
    print("Original dataset", len(original))
    sampled = df[df.purchase == 0].sample(n=len(original))
    concatenated = pd.concat([original, sampled])
    print("After downlsampling", len(concatenated))
    return concatenated


def add_schedule(df, path, future_index):
    # No promotions by default
    n_categories = df["j"].nunique()
    ps = pd.DataFrame({
        "j": np.arange(n_categories),
        "discount": np.zeros(n_categories),
        "advertised": np.zeros(n_categories),
    }).set_index("j")

    # There's no need to fill all promotions days
    ps.update(
        pd.read_csv(path / "promotion_schedule.csv").set_index("j")
    )
    ps = ps.to_dict()

    idx = df.t == future_index
    df.loc[idx, "advertised"] = df[idx]["j"].map(ps['advertised'])
    df.loc[idx, "price"] -= df[idx]["j"].map(ps['discount'])
    return df


def reduce_size(df):
    for col in ["i", "j", "t"]:
        df[col] = df[[col]].astype(np.uint16)

    df["price"] = df[["price"]].astype(np.float32)
    df["advertised"] = df[["advertised"]].astype(np.float32)
    return df


def prepare_dataset(path, future_index=49):
    df = pd.read_csv(path / "train.csv")

    # Create target variable
    df["purchase"] = True

    # Add empty bins
    df = extend_dataset(df)

    # Default types are too heavy for this dataset
    df = reduce_size(df)

    # Add schedule for the future prediction
    # extended = add_schedule(extended, path, future_index)
    return df
