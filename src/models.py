"""Model zoo: the five models from the papers plus a soft-voting ensemble."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL, NUMERIC, RANDOM_STATE as RS


def _prep() -> ColumnTransformer:
    # Fitted inside the Pipeline, so scaling/encoding never sees test folds (no leakage).
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def _svm(): return SVC(probability=True, class_weight="balanced", random_state=RS)
def _rf(): return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS, n_jobs=-1)
def _mlp(): return MLPClassifier((32, 16), max_iter=2000, random_state=RS)


def build_models() -> dict:
    clfs = {
        "SVM": _svm(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RS),
        "RandomForest": _rf(),
        "ANN-MLP": _mlp(),
        "Voting": VotingClassifier([("rf", _rf()), ("svm", _svm()), ("mlp", _mlp())], voting="soft"),
    }
    return {n: Pipeline([("prep", _prep()), ("clf", c)]) for n, c in clfs.items()}
