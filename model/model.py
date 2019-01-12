from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # noqa


# TODO: Remove me
def read_dataset():
    return None, None, None, None


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, records=False, pattern=None):
        self.columns = columns
        self.records = records
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")

        if self.pattern is not None:
            return X[[col for col in X.columns if self.pattern in col]]

        return X[self.columns]


def categorical(colname):
    return make_pipeline(
        PandasSelector(colname, records=True),
        DictVectorizer(),
    )


def build_model(classifier=LogisticRegression(solver="lbfgs")):
    model = make_pipeline(
        make_union(
            # categorical([
            #     "i",
            #     "j"
            # ]),
            make_pipeline(
                PandasSelector([
                    # "cum_price_i",
                    # "cum_price_j",
                    # "cum_ij",
                    # "cum_purchases_i",
                    # "cum_purchases_j",
                    "price",
                    "advertised",
                    "discount",
                    "weekday",
                    "days_passed_since_last_ad",
                    "t"
                ]),
                # StandardScaler()
            ),
            PandasSelector(pattern="j_"),
        ),
        classifier
    )
    return model
