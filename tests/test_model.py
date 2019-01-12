from model.model import build_model
from model.features import generate_features
from model.model_selection import xy


# Check if model agrees with the schema in data
def test_model_pipeline(data):
    dataset = generate_features(data.sample(100))
    X, y = xy(dataset)

    model = build_model()
    model.fit(X, y)
    print(dataset.info())
