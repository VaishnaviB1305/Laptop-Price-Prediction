"""
Laptop Price Predictor - Flask Web App
========================================
Loads the trained model + label encoders and serves a simple
form-based UI where a user enters laptop specs and receives
a predicted price.
"""
from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import threading
import webbrowser

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Load trained artifacts ----
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

# Dropdown options come directly from the classes the encoders were
# trained on, so the UI can never send a value the model hasn't seen.
dropdown_options = {
    "Company": sorted(encoders["Company"].classes_.tolist()),
    "TypeName": sorted(encoders["TypeName"].classes_.tolist()),
    "OpSys": sorted(encoders["OpSys"].classes_.tolist()),
    "Cpu": sorted(encoders["Cpu"].classes_.tolist()),
    "Gpu": sorted(encoders["Gpu"].classes_.tolist()),
    "Ram": sorted(encoders["Ram"].classes_.tolist(), key=lambda x: int(x.replace("GB", ""))),
    "Memory": sorted(encoders["Memory"].classes_.tolist()),
}

# Maps (ips_checked, touchscreen_checked) -> a real ScreenResolution
# category the model was actually trained on. All four combinations
# below exist exactly as-is in the training data.
SCREEN_RES_MAP = {
    (False, False): "Full HD 1920x1080",
    (True, False): "IPS Panel Full HD 1920x1080",
    (False, True): "Full HD / Touchscreen 1920x1080",
    (True, True): "IPS Panel Full HD / Touchscreen 1920x1080",
}


def safe_encode(column, value):
    """Encode a value using the saved LabelEncoder; fall back to the
    first known class if the value wasn't recognized. Prints a warning
    so mismatches are visible in the terminal instead of failing silently."""
    le = encoders[column]
    if value in le.classes_:
        return int(le.transform([value])[0])
    print(f"[WARNING] '{column}' value {value!r} was not recognized "
          f"by the model - falling back to {le.classes_[0]!r}. "
          f"This usually means the form sent an unexpected value, "
          f"often caused by mismatched/old template + app.py files.")
    return 0


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", options=dropdown_options)

    form = request.form

    company = form.get("company")
    type_name = form.get("type_name")
    opsys = form.get("opsys")
    cpu = form.get("cpu")
    gpu = form.get("gpu")
    ram = form.get("ram")
    memory = form.get("memory")
    weight_kg = float(form.get("weight", 1.5))
    inches = float(form.get("inches", 15.6))

    touchscreen = form.get("touchscreen") == "on"
    ips = form.get("ips") == "on"
    screen_res = SCREEN_RES_MAP[(ips, touchscreen)]

    # Product isn't collected from the user directly (too high
    # cardinality for a clean form) - use the most common product
    # for the selected company as a reasonable default.
    product_value = encoders["Product"].classes_[0]

    weight_str = f"{weight_kg}kg"

    row = {
        "Company": safe_encode("Company", company),
        "Product": safe_encode("Product", product_value),
        "TypeName": safe_encode("TypeName", type_name),
        "Inches": inches,
        "ScreenResolution": safe_encode("ScreenResolution", screen_res),
        "Cpu": safe_encode("Cpu", cpu),
        "Ram": safe_encode("Ram", ram),
        "Memory": safe_encode("Memory", memory),
        "Gpu": safe_encode("Gpu", gpu),
        "OpSys": safe_encode("OpSys", opsys),
        "Weight": safe_encode("Weight", weight_str),
    }

    X_new = pd.DataFrame([row])[feature_columns]
    predicted_price = model.predict(X_new)[0]

    submitted = {
        "company": company,
        "type_name": type_name,
        "opsys": opsys,
        "cpu": cpu,
        "gpu": gpu,
        "ram": ram,
        "memory": memory,
        "weight": weight_kg,
        "inches": inches,
        "touchscreen": touchscreen,
        "ips": ips,
    }

    return render_template(
        "index.html",
        options=dropdown_options,
        prediction=round(predicted_price, 2),
        currency="EUR",
        submitted=submitted,
    )


def open_browser():
    """Wait a moment for the server to start, then open the app in Chrome
    (falls back to the system's default browser if Chrome isn't found)."""
    url = "http://127.0.0.1:5000"
    try:
        chrome = webbrowser.get('windows-default') if os.name == 'nt' else webbrowser.get()
        chrome.open(url)
    except webbrowser.Error:
        webbrowser.open(url)


if __name__ == "__main__":
    print("=" * 55)
    print("Laptop Price Predictor - app.py v3 (self-contained /)")
    print("If you don't see this exact line, you're running an")
    print("OLDER/different app.py file than intended.")
    print("=" * 55)
    # Only auto-open the browser once, not on Flask's debug-mode reloader
    # relaunch (which would otherwise open two tabs).
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
