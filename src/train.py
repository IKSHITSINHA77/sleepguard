"""Train, compare (repeated stratified CV), evaluate on a hold-out set, save artefacts.

Usage: python -m src.train [--data PATH] [--keep-duplicates]
"""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

from . import config
from .config import BOUNDS, CATEGORICAL, DATA_PATH, FEATURES, RANDOM_STATE, TARGET
from .data import load_data
from .models import build_models


def main(data_path=DATA_PATH, keep_duplicates=False, out_dir=None) -> dict:
    model_dir = (out_dir / "models") if out_dir else config.MODEL_DIR
    report_dir = (out_dir / "reports") if out_dir else config.REPORT_DIR

    df = load_data(data_path, drop_duplicates=not keep_duplicates)
    counts = df[TARGET].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError(f"Need >=2 classes with >=3 rows each, got {counts.to_dict()}")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    splits = max(2, min(5, int(y_tr.value_counts().min())))   # edge case: tiny minority class
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=3, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision_macro",
               "recall": "recall_macro", "f1": "f1_macro"}

    rows = []
    for name, pipe in build_models().items():
        s = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model": name, **{m: round(s[f"test_{m}"].mean(), 4) for m in scoring},
                     "f1_std": round(s["test_f1"].std(), 4)})
    cv_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best = cv_df.loc[0, "model"]

    final = build_models()[best].fit(X_tr, y_tr)
    pred = final.predict(X_te)
    test_f1 = float(f1_score(y_te, pred, average="macro"))

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(report_dir / "cv_results.csv", index=False)
    (report_dir / "classification_report.txt").write_text(classification_report(y_te, pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, cmap="Blues", xticks_rotation=20)
    plt.title(f"{best} - hold-out confusion matrix"); plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150); plt.close()

    imp = permutation_importance(final, X_te, y_te, scoring="f1_macro", n_repeats=10,
                                 random_state=RANDOM_STATE)
    pd.Series(imp.importances_mean, index=FEATURES).sort_values().plot.barh(figsize=(7, 5), color="#3F4FBF")
    plt.title("Permutation importance (macro-F1 drop)")
    plt.tight_layout(); plt.savefig(report_dir / "feature_importance.png", dpi=150); plt.close()

    meta = {"best_model": best, "classes": list(final.classes_), "test_f1_macro": round(test_f1, 4),
            "n_rows": int(len(df)), "bounds": BOUNDS,
            "choices": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}}
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    joblib.dump(final, model_dir / "model.joblib")

    print(cv_df.to_string(index=False))
    print(f"\nBest: {best} | hold-out macro-F1 = {test_f1:.3f} | rows = {len(df)}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--keep-duplicates", action="store_true")
    a = ap.parse_args()
    main(a.data, a.keep_duplicates)
