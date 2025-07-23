import json
import pytest
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from src.train import get_trained_model

def test_config_file():
    with open("config/config.json") as f:
        config = json.load(f)
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

def test_model_type():
    model = get_trained_model()
    assert isinstance(model, Pipeline)

def test_model_accuracy():
    model = get_trained_model()
    X, y = load_digits(return_X_y=True)
    accuracy = model.score(X, y)
    assert accuracy >= 0.8, f"Accuracy too low: {accuracy:.4f}"

def test_model_f1_score():
    model = get_trained_model()
    X, y = load_digits(return_X_y=True)
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average="weighted")
    assert f1 >= 0.8, f"F1 Score too low: {f1:.4f}"