"""Central configuration: paths, column schema, validation bounds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Sleep Disorder"
NUMERIC = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
           "Stress Level", "Heart Rate", "Daily Steps", "Systolic", "Diastolic"]
CATEGORICAL = ["Gender", "Occupation", "BMI Category"]
FEATURES = NUMERIC + CATEGORICAL

REQUIRED_RAW = {"Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
                "Physical Activity Level", "Stress Level", "BMI Category",
                "Blood Pressure", "Heart Rate", "Daily Steps", "Sleep Disorder"}

# Inclusive plausibility bounds used to reject nonsense input at inference time.
BOUNDS = {
    "Age": (18, 100), "Sleep Duration": (2, 14), "Quality of Sleep": (1, 10),
    "Physical Activity Level": (0, 300), "Stress Level": (1, 10),
    "Heart Rate": (35, 200), "Daily Steps": (0, 40000),
    "Systolic": (70, 250), "Diastolic": (40, 150),
}
RANDOM_STATE = 42
