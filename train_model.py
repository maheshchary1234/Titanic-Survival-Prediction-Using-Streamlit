"""Train the Project 04 Titanic Decision Tree and save model.joblib.

Run this once before starting the Streamlit app:
    python train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
FEATURES = ["pclass", "sex", "age", "fare"]
TARGET = "survived"

def main():
    print("Downloading the Titanic dataset...")
    data = pd.read_csv(DATA_URL)
    data = data[["survived", *FEATURES]].dropna().copy()
    data["sex"] = (data["sex"] == "female").astype(int)
    data["survived"] = data["survived"].astype(int)

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Store metadata with the estimator so the app and model stay aligned.
    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "sex_encoding": {"male": 0, "female": 1},
        "model_name": "DecisionTreeClassifier(max_depth=4, random_state=42)",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved trained model artifact to: {MODEL_PATH}")
    print(f"Cleaned dataset rows: {len(data)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Test accuracy: {artifact['test_accuracy']:.3f}")
    print("Confusion matrix:")
    print(artifact["confusion_matrix"])

if __name__ == "__main__":
    main()
