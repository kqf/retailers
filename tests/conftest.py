import pytest
from model.data import prepare_dataset
from model.mc import generate_dataset, generate_schedule
from pathlib import Path

DATA_PATH = Path("data/")


@pytest.fixture(scope="session")
def data_path():
    train_path = DATA_PATH / "train.csv"
    schedule_path = DATA_PATH / "promotion_schedule.csv"

    files_exist = train_path.is_file() and schedule_path.is_file()

    if not files_exist:
        generate_dataset(train_path)
        generate_schedule(schedule_path)

    yield DATA_PATH

    # teardown
    # if not files_exist:
    #     train_path.unlink()
    #     schedule_path.unlink()


@pytest.fixture(scope="session")
def data(data_path):
    return prepare_dataset(data_path)
