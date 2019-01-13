import numpy as np
from tqdm import tqdm


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


# NB: Feature genereation should be a part of a model (fit/transform methods).
#     Here it's implented as a separate module in order to have a bit more
#     flexibility (for model selection) and it's easier to read.


def generate_features(df):
    df = df.sort_values(by=["t"]).reset_index(drop=True)
    features = [
        lambda x: time_since_last_ad(x),
        lambda x: discount(x),
    ]

    for f in tqdm(features):
        df = f(df)

    df["weekday"] = df["t"].apply(lambda x: x % 7)
    return df
