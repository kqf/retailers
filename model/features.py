import numpy as np
import pandas as pd
from tqdm import tqdm


def shifted_sum(x, dtype=np.float32):
    return x.shift().cumsum().fillna(0)


def cumsum(df, target, by, out_col):
    df[out_col] = df.groupby(by)[target].transform(
        lambda x: x.cumsum().shift().fillna(0))
    return df


def time_since_last_ad(df):
    df["days_passed_since_last_ad"] = df["t"] - df["t"].where(
        df["advertised"] > 0).groupby(df["j"]).ffill()
    df["days_passed_since_last_ad"].fillna(-9, inplace=True)
    return df


def discount(df):
    prod = df.groupby("j")["price"]
    df["discount"] = (prod.transform(
        lambda x: float(x.mode())) - df['price']) > 0
    df["discount"] = df["discount"].astype(np.int8)
    return df


def one_hot_purchase(df, column):
    one_hot = pd.get_dummies(df[column], prefix="j")
    one_hot = one_hot.mul(df["purchase"], axis=0)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df, one_hot


def cum_j(df, col="j"):
    cum, one_hot = one_hot_purchase(df, col)
    one_hot_cum = cum.groupby("i").transform(
        lambda x: x.shift().cumsum().fillna(0))[one_hot.columns] * one_hot
    return df.join(one_hot_cum)

# NB: Feature genereation should be a part of a model (fit/transform methods).
#     Here it's implented as a separate module in order to have a bit more
#     flexibility (for model selection) and it's easier to read.


def generate_features(df):
    df = df.sort_values(by=["t"]).reset_index(drop=True)
    features = [
        lambda x: time_since_last_ad(x),
        lambda x: cumsum(x, "purchase", "i", "cum_purchases_i"),
        lambda x: cumsum(x, "purchase", "j", "cum_purchases_j"),
        lambda x: cumsum(x, "price", "i", "cum_price_i"),
        lambda x: cumsum(x, "price", "j", "cum_price_j"),
        lambda x: cumsum(x, "purchase", ["i", "j"], "cum_ij"),
        lambda x: cum_j(x),
        lambda x: discount(x),
    ]

    for f in tqdm(features):
        df = f(df)

    df["weekday"] = df["t"].apply(lambda x: x % 7)
    return df
