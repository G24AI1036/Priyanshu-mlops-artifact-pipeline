import pickle
from sklearn.datasets import load_digits
from sklearn.metrics import log_loss, f1_score
import os

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    loss = log_loss(y, y_proba)
    f1 = f1_score(y, y_pred, average="weighted") 
    return y_pred, loss, f1

def main():
    model_path = "model/train.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)

    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target

    y_pred, loss, f1 = evaluate_model(model, X, y)

    print(f"predictions: {y_pred[:10]}")
    print(f"log Loss: {loss:.4f}")
    print(f"f1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()