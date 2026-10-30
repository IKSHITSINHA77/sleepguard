"""Generate a SYNTHETIC dataset with the Kaggle schema (smoke tests only, not real data)."""
import sys

import numpy as np
import pandas as pd

OCC = ["Nurse", "Doctor", "Engineer", "Lawyer", "Teacher", "Salesperson", "Accountant"]


def make(n=400, seed=42) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    bmi = r.choice(["Normal", "Overweight", "Obese"], n, p=[.5, .4, .1])
    sleep = np.round(r.uniform(5.5, 8.5, n), 1)
    stress = r.integers(3, 9, n)
    sys_ = r.integers(115, 145, n) + 5 * (bmi != "Normal")
    dia = sys_ - r.integers(30, 45, n)
    p_apnea = .05 + .5 * ((bmi != "Normal") & (sys_ > 130))
    p_ins = .05 + .5 * ((sleep < 6.5) & (stress > 6))
    u = r.random(n)
    dis = np.where(u < p_apnea, "Sleep Apnea", np.where(u < p_apnea + p_ins, "Insomnia", ""))
    quality = np.clip(np.round(10 - stress * .6 + r.normal(0, 1, n)), 1, 10).astype(int)
    return pd.DataFrame({
        "Person ID": range(1, n + 1), "Gender": r.choice(["Male", "Female"], n),
        "Age": r.integers(27, 60, n), "Occupation": r.choice(OCC, n),
        "Sleep Duration": sleep, "Quality of Sleep": quality,
        "Physical Activity Level": r.integers(30, 90, n), "Stress Level": stress,
        "BMI Category": bmi, "Blood Pressure": [f"{a}/{b}" for a, b in zip(sys_, dia)],
        "Heart Rate": r.integers(65, 85, n), "Daily Steps": r.integers(3000, 10000, n),
        "Sleep Disorder": dis,
    })


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_sleep.csv"
    make().to_csv(out, index=False)
    print("wrote", out)
