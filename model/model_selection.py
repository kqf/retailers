import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import BaseCrossValidator
from model.data import downsample


def xy(df):
    return df.drop(columns="purchase"), df["purchase"].values


def train_last_day_split(df, day, full_training=False):
    train = downsample(df[df["t"] < day], skip=full_training)
    return xy(train), xy(df[df["t"] == day])


class HomogenousSplitCV(BaseCrossValidator):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None):
        start, stop = X.t.min(), X.t.max()
        step = (stop - start) // self.n_splits or 1
        for i in range(start + step, stop, step):
            yield X.t < i, X.t > i


class LastDaysSplitCV(BaseCrossValidator):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None):
        start, stop = X.t.min(), X.t.max()
        assert stop - start > self.n_splits
        for i in range(stop - self.n_splits + 1, stop + 1):
            yield X.t < i, X.t == i


def plot_roc_curve(y, y_pred, name):
    fpr, tpr, _ = roc_curve(y, y_pred)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver operating characteristic')
    return plt.plot(fpr, tpr, label=name + ' AUC = {}'.format(roc_auc))
