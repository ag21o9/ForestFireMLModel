# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import datetime

# ----------------------
# Config / load models
# ----------------------
MODEL_PATH = "fire_risk_rf_1to10.pkl"
MONTH_ENCODER_PATH = "month_encoder.pkl"
DAY_ENCODER_PATH = "day_encoder.pkl"

model = joblib.load(MODEL_PATH)
le_month = joblib.load(MONTH_ENCODER_PATH)
le_day = joblib.load(DAY_ENCODER_PATH)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests for frontend during hackathon

# ----------------------
# Helpers
# ----------------------
def estimate_fire_indexes(temp, rh, wind, rain):
    """
    Quick heuristic to estimate FFMC, DMC, DC, ISI from basic weather.
    These are **approximate** and meant for a demo/prototype.
    """
    # Clamp inputs to dataset-like ranges
    temp = float(np.clip(temp,  -10, 60))
    rh   = float(np.clip(rh,   0, 100))
    wind = float(np.clip(wind, 0, 50))
    rain = float(np.clip(rain, 0, 500))

    # Heuristics (simple, demo-purpose)
    FFMC = np.clip(20 + (temp * 1.6) - (rh * 0.15) + (wind * 0.6) - (rain * 0.1), 18.7, 96.2)
    DMC  = np.clip(1 + (100 - rh) * (temp/30) + wind*0.5 - rain*0.2, 1.1, 291.3)
    DC   = np.clip(10 + (100 - rh) * (temp/20) + wind*0.7 - rain*0.05, 7.9, 860.6)
    ISI  = np.clip((wind * 0.8) + (FFMC / 30.0), 0.0, 56.1)

    return float(round(FFMC, 2)), float(round(DMC, 2)), float(round(DC, 2)), float(round(ISI, 2))


def score_to_bucket(score):
    s = int(score)
    if s <= 3:
        return "Low", "green"
    elif s <= 6:
        return "Medium", "yellow"
    elif s <= 8:
        return "High", "orange"
    else:
        return "Extreme", "red"


def encode_month_day(month, day):
    """
    Accept month/day either as strings ('jan','mon') or already-encoded ints.
    """
    # month transform
    if isinstance(month, str):
        month_enc = int(le_month.transform([month])[0])
    else:
        month_enc = int(month)
    # day transform
    if isinstance(day, str):
        day_enc = int(le_day.transform([day])[0])
    else:
        day_enc = int(day)
    return month_enc, day_enc

# ----------------------
# API Endpoints
# ----------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON:
    Required fields: X (1-9) or lat/lon mapping handled by frontend, Y (2-9)
    Either provide FFMC,DMC,DC,ISI OR provide temp,RH,wind,rain and month/day.
    Example request body:
    {
      "X": 4,
      "Y": 5,
      "month": "aug",
      "day": "sun",
      "temp": 33.1,
      "RH": 20,
      "wind": 6.7,
      "rain": 0.0
    }
    OR provide FFMC,DMC,DC,ISI directly.
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    # Basic required X,Y
    X_val = int(data.get("X", 5))
    Y_val = int(data.get("Y", 5))

    # date / month / day (defaults to today's date if not provided)
    month = data.get("month", None)
    day   = data.get("day", None)
    if month is None or day is None:
        now = datetime.datetime.utcnow()
        # convert month number to short string used by encoder (encoder was trained on textual months like 'aug')
        # The encoder expects the same labels as training set; assume english abbreviations used earlier.
        # If month encoder mapping is unknown, passing numeric will attempt to use it as integer index.
        month = now.strftime("%b").lower()  # e.g. 'Nov' -> 'nov'
        day   = now.strftime("%a").lower()  # e.g. 'Fri' -> 'fri'

    # If encoders don't know the exact lowercase format, ensure matching to trained labels:
    # We'll attempt transform; if it fails, user should send encoded month/day ints instead.
    try:
        month_enc, day_enc = encode_month_day(month, day)
    except Exception as e:
        return jsonify({"error": "Month/day encoding failed. Provide month/day as the same strings used during training (e.g. 'aug','sun') or numeric encodings.", "details": str(e)}), 400

    # If FFMC,DMC,DC,ISI were provided use them else estimate from weather
    ffmc = data.get("FFMC", None)
    dmc  = data.get("DMC", None)
    dc   = data.get("DC", None)
    isi  = data.get("ISI", None)

    if None in (ffmc, dmc, dc, isi):
        # Need temp, RH, wind, rain to estimate indexes
        try:
            temp = float(data.get("temp"))
            rh   = float(data.get("RH"))
            wind = float(data.get("wind"))
            rain = float(data.get("rain", 0.0))
        except Exception as e:
            return jsonify({"error": "Missing meteorological inputs for index estimation. Provide either FFMC,DMC,DC,ISI OR provide temp,RH,wind,rain.", "details": str(e)}), 400

        ffmc, dmc, dc, isi = estimate_fire_indexes(temp, rh, wind, rain)

    # Build ordered feature vector in same order used during training:
    # ['X','Y','month_enc','day_enc','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
    temp_val = float(data.get("temp", 20.0))
    rh_val   = float(data.get("RH", 40.0))
    wind_val = float(data.get("wind", 1.0))
    rain_val = float(data.get("rain", 0.0))

    features = [
        float(X_val),
        float(Y_val),
        int(month_enc),
        int(day_enc),
        float(ffmc),
        float(dmc),
        float(dc),
        float(isi),
        float(temp_val),
        float(rh_val),
        float(wind_val),
        float(rain_val)
    ]

    # Predict
    try:
        pred = model.predict([features])[0]
        pred = int(pred)
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    bucket, color = score_to_bucket(pred)

    return jsonify({
        "score": pred,
        "bucket": bucket,
        "color": color,
        "features_used": {
            "X": X_val, "Y": Y_val,
            "month_enc": int(month_enc), "day_enc": int(day_enc),
            "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi,
            "temp": temp_val, "RH": rh_val, "wind": wind_val, "rain": rain_val
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
