from model.model import build_model
from model.model_selection import xy


# Check if model agrees with the schema in data
def test_model_pipeline(data):
    X, y = xy(data[data.t < 48].sample(10000))
    model = build_model()
    model.fit(X, y)
