from pathlib import Path

import click
from model.model import build_model
from model.data import prepare_dataset
from model.model_selection import train_last_week_split


@click.command()
@click.option('--data_path',
              type=Path,
              help='Path to the dataset',
              required=True)
@click.option('--week',
              default=49,
              type=int,
              help='Week number to predict',
              required=True)
def main(data_path, week):
    data = prepare_dataset(data_path)
    (X_tr, y_tr), (X_te, y_te) = train_last_week_split(
        data, week=week, full_training=False)
    clf = build_model()
    clf.fit(X_tr, y_tr)

    submission = X_te.loc[:, ["i", "j"]]
    submission["prediction"] = clf.predict_proba(X_te)[:, 1]
    submission.to_csv("prediction.csv", index=False)
