import numpy as np
import pandas as pd
from model.data import outer_product

# NB: There is no modeling here, this is functions generates dummy dataset
#     to check for data preprocessing, possible leaks and for CI tests


def generate_dataset(file_path, positive_size=80000, min_price=0.2,
                     max_price=3.5, ad_p=0.5, days=49):
    dummy = pd.DataFrame(list(outer_product(days=days)),
                         columns=["i", "j", "t"])

    # Generate prices
    dummy["price"] = dummy.groupby(["j", "t"])["i"].transform(
        lambda x: np.random.uniform(min_price, max_price))

    # Generate advertisements
    dummy["advertised"] = dummy.groupby(["j", "t"])["i"].transform(
        lambda x: np.random.binomial(p=ad_p, n=1))

    # Generate purchases
    purchased = dummy.sample(n=positive_size)
    purchased.sort_values(by=["t"], inplace=True)
    purchased.to_csv(file_path, index=False)


def generate_schedule(file_path, size=40):
    targets, promoted = np.arange(size), np.random.randint(0, size)
    empty = pd.DataFrame({
        "j": targets,
        "advertised": np.zeros_like(targets),
        "discount": np.zeros_like(targets),
    })
    empty.loc[empty["j"] == promoted, "advertised"] = np.random.uniform()
    empty.loc[empty["j"] == promoted, "discount"] = np.random.uniform()
    empty.to_csv(file_path, index=False)
