import pytest

from scripts.make_synthetic import make
from src.data import load_data
from src.predict import Predictor
from src.train import main as train_main

GOOD = {"Age": 40, "Sleep Duration": 6.5, "Quality of Sleep": 6, "Physical Activity Level": 45,
        "Stress Level": 6, "Heart Rate": 72, "Daily Steps": 6000, "Systolic": 130,
        "Diastolic": 85, "Gender": "Male", "Occupation": "Nurse", "BMI Category": "Normal"}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("d")
    csv = d / "s.csv"
    make().to_csv(csv, index=False)
    meta = train_main(csv, out_dir=d)
    return Predictor(d / "models"), meta, csv


def test_load_cleans_schema(trained):
    df = load_data(trained[2])
    assert {"Systolic", "Diastolic"} <= set(df) and "Blood Pressure" not in df
    assert not df.isna().any().any() and df.duplicated().sum() == 0


def test_missing_file_and_columns(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_data(tmp_path / "bad.csv")


def test_predict_ok(trained):
    out = trained[0].predict(GOOD)
    assert out["prediction"] in trained[1]["classes"]
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3


@pytest.mark.parametrize("patch", [
    {"Age": 5}, {"Age": "abc"}, {"Systolic": 80, "Diastolic": 90},
    {"Occupation": "Wizard"}, {"Heart Rate": float("nan")}])
def test_predict_rejects_bad_input(trained, patch):
    with pytest.raises(ValueError):
        trained[0].predict({**GOOD, **patch})


def test_missing_field_rejected(trained):
    with pytest.raises(ValueError):
        trained[0].predict({k: v for k, v in GOOD.items() if k != "Age"})
