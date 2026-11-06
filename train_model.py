"""
train_model.py - Train and compare models for diabetes risk screening.

Dataset: CDC BRFSS 2015 Diabetes Health Indicators (Kaggle: alexteboul/diabetes-health-indicators-dataset)
File:    diabetes_binary_5050split_health_indicators_BRFSS2015.csv  (~70,700 rows, 50/50 balanced)

What it does:
  1. Loads the CSV (from ./data/ or downloads it with kagglehub)
  2. Splits it once into 80% train / 20% test (same split for every model)
  3. Trains Logistic Regression, Random Forest, XGBoost, LightGBM
  4. Reports Accuracy, Precision, Recall, F1, AUC + confusion matrix for each
  5. Tunes the best model (by Recall), retrains it on all data
  6. Saves model + feature order + metrics to diabetes_model.pkl (used by app.py)

Run:  python train_model.py
"""

import glob
import json
import os
import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_FILE = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"
SEED = 42
OUT_MODEL = "diabetes_model.pkl"
OUT_METRICS = "model_results.json"


# ---------------------------------------------------------------- 1. Load data
def load_data() -> pd.DataFrame:
    local = os.path.join("data", DATA_FILE)
    if os.path.exists(local):
        path = local
    elif os.path.exists(DATA_FILE):
        path = DATA_FILE
    else:
        print("CSV not found in ./data/ - trying to download from Kaggle with kagglehub...")
        try:
            import kagglehub
            folder = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
            path = glob.glob(os.path.join(folder, "**", DATA_FILE), recursive=True)[0]
        except Exception as e:
            raise SystemExit(
                f"\nCould not find or download the dataset ({e}).\n"
                f"Download it from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset\n"
                f"and put {DATA_FILE} inside a folder called 'data' next to this script."
            )
    df = pd.read_csv(path).astype(int)
    print(f"Loaded {path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Class balance:", df[TARGET].value_counts(normalize=True).round(3).to_dict())
    return df


# ------------------------------------------------------------- 2. Models
def get_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                                 eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   random_state=SEED, verbose=-1),
    }


TUNING_GRIDS = {
    "Logistic Regression": {"logisticregression__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"n_estimators": [200, 400], "max_depth": [8, 12, None],
                      "min_samples_leaf": [1, 5, 10]},
    "XGBoost": {"n_estimators": [200, 400], "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0]},
    "LightGBM": {"n_estimators": [200, 400], "num_leaves": [15, 31, 63],
                 "learning_rate": [0.03, 0.05, 0.1]},
}


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "AUC": round(roc_auc_score(y_test, prob), 4),
        "ConfusionMatrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


# ------------------------------------------------------------------ main
def main():
    df = load_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    features = list(X.columns)

    # Same split for every model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}\n")

    # 3-4. Train and compare
    models, results = get_models(), {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        results[name]["TrainSeconds"] = round(time.time() - t0, 1)
        print(f"Trained {name:<20} in {results[name]['TrainSeconds']}s")

    table = pd.DataFrame({n: {k: v for k, v in r.items() if k != "ConfusionMatrix"}
                          for n, r in results.items()}).T.sort_values("Recall", ascending=False)
    print("\n=== Model comparison (test set, sorted by Recall) ===")
    print(table.to_string())
    print("\n=== Confusion matrices ===")
    for name, r in results.items():
        cm = r["ConfusionMatrix"]
        print(f"{name:<20} TN={cm['TN']:>6}  FP={cm['FP']:>6}  FN={cm['FN']:>6}  TP={cm['TP']:>6}")

    # 5. Tune the best model (by Recall, tie-break F1)
    best_name = table.sort_values(["Recall", "F1"], ascending=False).index[0]
    print(f"\nBest model by Recall: {best_name} -> tuning...")
    search = RandomizedSearchCV(get_models()[best_name], TUNING_GRIDS[best_name],
                                n_iter=10, scoring="recall", cv=3,
                                random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    tuned_metrics = evaluate(search.best_estimator_, X_test, y_test)
    print("Best params:", search.best_params_)
    print("Tuned test metrics:", {k: v for k, v in tuned_metrics.items() if k != "ConfusionMatrix"})

    # Keep tuned version only if it is at least as good on Recall
    if tuned_metrics["Recall"] >= results[best_name]["Recall"]:
        final_params, final_metrics = search.best_params_, tuned_metrics
        final_model = search.best_estimator_
    else:
        print("Tuning did not improve Recall - keeping the original settings.")
        final_params, final_metrics = {}, results[best_name]
        final_model = get_models()[best_name]

    # 6. Retrain on 100% of the data and save
    final_model.fit(X, y)
    bundle = {
        "model": final_model,
        "model_name": best_name,
        "features": features,              # exact column order the model expects
        "defaults": X.median().astype(int).to_dict(),
        "test_metrics": final_metrics,
        "params": final_params,
    }
    joblib.dump(bundle, OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump({"comparison": results, "best_model": best_name,
                   "best_params": final_params, "final_test_metrics": final_metrics}, f, indent=2)

    print(f"\nSaved {OUT_MODEL} ({best_name}) and {OUT_METRICS}")

    # Quick sanity check: load it back and score two example people
    loaded = joblib.load(OUT_MODEL)
    examples = pd.DataFrame([loaded["defaults"], loaded["defaults"]])[loaded["features"]]
    examples.loc[0, ["Age", "BMI", "GenHlth", "HighBP", "HighChol"]] = [3, 22, 1, 0, 0]
    examples.loc[1, ["Age", "BMI", "GenHlth", "HighBP", "HighChol"]] = [11, 38, 5, 1, 1]
    risk = loaded["model"].predict_proba(examples)[:, 1]
    print(f"Sanity check -> healthy 30-year-old: {risk[0]:.0%} risk | 70-year-old, BMI 38, poor health: {risk[1]:.0%} risk")


if __name__ == "__main__":
    main()
