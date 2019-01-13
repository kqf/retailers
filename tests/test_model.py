from model.model import build_model
from model.model_selection import xy


# Check if model agrees with the schema in data
def test_model_pipeline(data):
    X, y = xy(data.sample(1000))
    model = build_model()
    model.fit(X, y)
