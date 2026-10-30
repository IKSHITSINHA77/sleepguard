"""Validated inference wrapper used by the web app."""
import json
import math

import joblib
import pandas as pd

from .config import BOUNDS, CATEGORICAL, FEATURES, MODEL_DIR, NUMERIC


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        try:
            self.model = joblib.load(model_dir / "model.joblib")
            self.meta = json.loads((model_dir / "meta.json").read_text())
        except FileNotFoundError as e:
            raise RuntimeError("No trained model found. Run: python -m src.train") from e

    def validate(self, payload: dict) -> pd.DataFrame:
        """Return a 1-row frame or raise ValueError listing every problem at once."""
        errors, row = [], {}
        for col in NUMERIC:
            try:
                v = float(payload[col])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{col} must be a number"); continue
            lo, hi = BOUNDS[col]
            if math.isnan(v) or not lo <= v <= hi:
                errors.append(f"{col} must be between {lo} and {hi}")
            row[col] = v
        for col in CATEGORICAL:
            v = str(payload.get(col, "")).strip()
            if v not in self.meta["choices"][col]:
                errors.append(f"{col} must be one of {self.meta['choices'][col]}")
            row[col] = v
        if not errors and row["Systolic"] <= row["Diastolic"]:
            errors.append("Systolic pressure must be higher than diastolic")
        if errors:
            raise ValueError("; ".join(errors))
        return pd.DataFrame([row])[FEATURES]

    def predict(self, payload: dict) -> dict:
        proba = self.model.predict_proba(self.validate(payload))[0]
        probs = {c: round(float(p), 4) for c, p in zip(self.model.classes_, proba)}
        label = max(probs, key=probs.get)
        return {"prediction": label, "confidence": probs[label], "probabilities": probs}
