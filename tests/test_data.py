import numpy as np


def test_has_same_price_in_time(data):
    price_recalculated = data.groupby(["j", "t"])["price"].transform(
        lambda x: x.mean())
    np.testing.assert_almost_equal(
        data["price"].values,
        price_recalculated.values)


def test_has_same_ads_time(data):
    price_recalculated = data.groupby(["j", "t"])["advertised"].transform(
        lambda x: x.mean())
    np.testing.assert_almost_equal(
        data["advertised"].values,
        price_recalculated.values)
