"""Build features, train classifiers, and save the model and metrics.

Usage:
    python -m src.train              # uses CWRU files if present, otherwise synthetic data
    python -m src.train --synthetic  # force synthetic data

Split strategy: windows from the same recording overlap and are highly
correlated, so a random split leaks information and inflates accuracy.
Instead, the first 70% of every recording is used for training and the
last 30% for testing (a time-based split, like predicting the future).
"""
import json
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLASSES, MODEL_DIR, PROCESSED_DIR
from src.data import load_recordings, windows
from src.features import FEATURE_COLS, build_table

TRAIN_FRACTION = 0.7


def main(prefer_real: bool = True) -> dict:
    recs = load_recordings(prefer_real=prefer_real)
    source = recs[0].source
    print(f"Loaded {len(recs)} recordings ({source})")

    df = build_table(recs, windows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"Feature table: {df.shape[0]} windows x {len(FEATURE_COLS)} features")

    train = df[df.position < TRAIN_FRACTION]
    test = df[df.position >= TRAIN_FRACTION]
    X_tr, y_tr = train[FEATURE_COLS], train.label
    X_te, y_te = test[FEATURE_COLS], test.label

    models = {
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=0,
                                                n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {"accuracy": accuracy_score(y_te, pred),
                         "macro_f1": f1_score(y_te, pred, average="macro")}
        print(f"{name:>20}: accuracy={results[name]['accuracy']:.3f} "
              f"macro_f1={results[name]['macro_f1']:.3f}")

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best = models[best_name]
    pred = best.predict(X_te)
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_te, pred, labels=CLASSES, zero_division=0))

    importances = None
    rf = models["random_forest"]
    importances = dict(sorted(zip(FEATURE_COLS, rf.feature_importances_.round(4).tolist()),
                              key=lambda kv: -kv[1]))

    metrics = {
        "data_source": source,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "models": results,
        "best_model": best_name,
        "classes": CLASSES,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=CLASSES).tolist(),
        "feature_importance": importances,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": FEATURE_COLS, "classes": CLASSES},
                MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {MODEL_DIR}")
    return metrics


if __name__ == "__main__":
    np.random.seed(0)
    main(prefer_real="--synthetic" not in sys.argv)
