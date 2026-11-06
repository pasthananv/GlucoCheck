"""
app.py - Web backend for the diabetes risk screening page.

Run:   python app.py      then open http://localhost:5000
Needs: diabetes_model.pkl (made by train_model.py) and index.html in the same folder.
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(HERE, "diabetes_model.pkl"))
MODEL, FEATURES, DEFAULTS = bundle["model"], bundle["features"], bundle["defaults"]
print(f"Loaded model: {bundle['model_name']} ({len(FEATURES)} features)")

app = Flask(__name__)


@app.route("/")
def home():
    return send_from_directory(HERE, "index.html")


def precautions(a: dict, level: str) -> list:
    """Personalised advice based on the person's answers."""
    tips = []
    if level == "high":
        tips.append("See a doctor and ask for a fasting blood sugar or HbA1c test within the next few weeks.")
    elif level == "moderate":
        tips.append("Ask for a blood sugar (HbA1c) test at your next check-up.")
    else:
        tips.append("Keep up your healthy habits and repeat this check every year.")
    if a["BMI"] >= 30:
        tips.append("Your BMI is in the obese range. Losing even 5-7% of body weight lowers diabetes risk a lot.")
    elif a["BMI"] >= 25:
        tips.append("Your BMI is above the healthy range. Aim for gradual weight loss through diet and activity.")
    if a["PhysActivity"] == 0:
        tips.append("Try to be active at least 150 minutes per week, e.g. a 30-minute brisk walk, 5 days a week.")
    if a["HighBP"] == 1:
        tips.append("Keep your blood pressure under control: take prescribed medicine and cut down on salt.")
    if a["HighChol"] == 1:
        tips.append("Limit fried and fatty foods and check your cholesterol regularly.")
    if a["Smoker"] == 1:
        tips.append("If you still smoke, quitting lowers your risk of diabetes complications and heart disease.")
    if a["HvyAlcoholConsump"] == 1:
        tips.append("Cut down on alcohol; heavy drinking raises blood sugar and weight.")
    if a["Fruits"] == 0 or a["Veggies"] == 0:
        tips.append("Eat vegetables and fruit every day, and swap sugary drinks for water.")
    tips.append("Warning signs to see a doctor sooner: strong thirst, frequent urination, blurry vision, "
                "slow-healing wounds, or unexplained weight loss.")
    return tips


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    answers = dict(DEFAULTS)
    try:
        for f in FEATURES:
            if f in data:
                answers[f] = int(round(float(data[f])))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid answer values"}), 400

    row = pd.DataFrame([answers])[FEATURES]
    risk = float(MODEL.predict_proba(row)[0, 1])
    level = "high" if risk >= 0.6 else "moderate" if risk >= 0.4 else "low"
    return jsonify({
        "risk": round(risk, 4),
        "level": level,
        "precautions": precautions(answers, level),
        "model": bundle["model_name"],
    })


def free_port(preferred=(8000, 8080, 8888, 5050)) -> int:
    """Return the first preferred port that is free, otherwise any free port."""
    import socket
    for p in preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    import threading
    import webbrowser

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 50 + f"\n  Open this in your browser:  {url}\n" + "=" * 50 + "\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)