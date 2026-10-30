"""SleepGuard Flask application with SQLite-backed user accounts."""
import json
import os
import secrets
import sqlite3
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from src.predict import Predictor

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SLEEPGUARD_SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SLEEPGUARD_HTTPS", "0") == "1",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
)
ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
DB_PATH = ROOT / "sleepguard_users.db"
_predictor = None

ADVICE = {
    "None": "The model did not identify a disorder pattern in the supplied data.",
    "Insomnia": "The supplied pattern was classified as resembling insomnia.",
    "Sleep Apnea": "The supplied pattern was classified as resembling sleep apnea.",
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()


init_db()


def csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def inject_user():
    user = None
    if session.get("user_id"):
        with get_db() as db:
            row = db.execute("SELECT id, name, email FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        if row:
            user = dict(row)
        else:
            session.clear()
    return {"current_user": user, "csrf_token": csrf_token()}


def require_csrf():
    expected = session.get("_csrf_token", "")
    supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
    return bool(expected and supplied and secrets.compare_digest(expected, supplied))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify(error="Please sign in to continue.", login_required=True), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


@app.get("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("auth.html", mode="login", error=None)


@app.post("/login")
def login_post():
    if not require_csrf():
        return render_template("auth.html", mode="login", error="Your session expired. Refresh the page and try again."), 400
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    with get_db() as db:
        user = db.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return render_template("auth.html", mode="login", error="Email or password is incorrect."), 401
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True
    csrf_token()
    next_url = request.args.get("next", "")
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = url_for("index")
    return redirect(next_url)


@app.get("/register")
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("auth.html", mode="register", error=None)


@app.post("/register")
def register_post():
    if not require_csrf():
        return render_template("auth.html", mode="register", error="Your session expired. Refresh the page and try again."), 400
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")
    if len(name) < 2 or len(name) > 80:
        error = "Please enter a name between 2 and 80 characters."
    elif len(email) > 254 or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        error = "Please enter a valid email address."
    elif len(password) < 8:
        error = "Password must contain at least 8 characters."
    elif password != confirm:
        error = "The passwords do not match."
    else:
        error = None
    if error:
        return render_template("auth.html", mode="register", error=error), 400
    try:
        with get_db() as db:
            cursor = db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            db.commit()
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return render_template("auth.html", mode="register", error="An account with this email already exists."), 409
    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    csrf_token()
    return redirect(url_for("index"))


@app.post("/logout")
@login_required
def logout():
    if not require_csrf():
        return "Invalid security token. Refresh and try again.", 400
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index():
    try:
        model_meta = predictor().meta
        setup_error = None
    except RuntimeError as exc:
        model_meta = None
        setup_error = str(exc)
    return render_template("index.html", meta=model_meta, setup_error=setup_error)


@app.post("/api/predict")
@login_required
def api_predict():
    if not require_csrf():
        return jsonify(error="Security token expired. Refresh the page and try again."), 400
    try:
        out = predictor().predict(request.get_json(silent=True) or {})
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400
    out["message"] = ADVICE.get(out["prediction"], "Review this result with a qualified professional.")
    out["model"] = predictor().meta.get("best_model", "Unknown")
    return jsonify(out)


@app.get("/api/analytics")
@login_required
def api_analytics():
    result = {"available": False, "models": [], "meta": {}, "reports": {}}
    model_meta_path = ROOT / "models" / "meta.json"
    cv_path = REPORT_DIR / "cv_results.csv"
    if model_meta_path.exists():
        result["meta"] = json.loads(model_meta_path.read_text(encoding="utf-8"))
    if cv_path.exists():
        df = pd.read_csv(cv_path).replace({float("nan"): None})
        result["models"] = df.to_dict(orient="records")
        result["available"] = True
    for filename in ("confusion_matrix.png", "feature_importance.png", "classification_report.txt"):
        if (REPORT_DIR / filename).exists():
            result["reports"][filename] = f"/reports/{filename}"
    return jsonify(result)


@app.get("/reports/<path:filename>")
@login_required
def report_file(filename):
    allowed = {"confusion_matrix.png", "feature_importance.png", "classification_report.txt"}
    if filename not in allowed:
        return jsonify(error="Report not found"), 404
    from flask import send_from_directory
    return send_from_directory(REPORT_DIR, filename)


@app.get("/health")
def health():
    try:
        return jsonify(status="ok", model=predictor().meta.get("best_model"))
    except RuntimeError as exc:
        return jsonify(status="model_not_trained", message=str(exc)), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
