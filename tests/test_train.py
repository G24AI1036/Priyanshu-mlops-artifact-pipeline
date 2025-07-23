import json
import pytest
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from src.train import get_trained_model

@pytest.fixture(scope="module")
def config():
    with open("config/config.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def model():
    return get_trained_model()

@pytest.fixture(scope="module")
def digits_data():
    return load_digits(return_X_y=True)

def test_config_file(config):
    assert "C" in config, "Missing 'C' in config"
    assert isinstance(config["C"], float), "'C' must be a float"

    assert "solver" in config, "Missing 'solver' in config"
    assert isinstance(config["solver"], str), "'solver' must be a string"

    assert "max_iter" in config, "Missing 'max_iter' in config"
    assert isinstance(config["max_iter"], int), "'max_iter' must be an integer"

def test_model_type(model):
    assert isinstance(model, Pipeline), "Model should be a scikit-learn Pipeline"

def test_model_accuracy(model, digits_data):
    X, y = digits_data
    accuracy = model.score(X, y)
    assert accuracy >= 0.8, f"Accuracy too low: {accuracy:.4f}"

def test_model_f1_score(model, digits_data):
    X, y = digits_data
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average="weighted")
    assert f1 >= 0.8, f"F1 Score too low: {f1:.4f}"
