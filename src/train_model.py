# src/train_model.py
import os
import joblib
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

from load_data import load_csv

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, 'model.joblib')

def train_and_save(model_path=MODEL_PATH):
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_csv()

    print("Training XGBoost classifier (this may take a minute)...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    print("Saving model to:", model_path)
    joblib.dump(model, model_path)
    print("Saved.")

if __name__ == '__main__':
    train_and_save()
