from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import json
from pathlib import Path
import io

# ================= APP INIT =================
app = Flask(__name__)

# ================= PROJECT ROOT =================
PROJECT_ROOT = Path(__file__).parent

# ================= PATHS =================
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "models" / "current_model.joblib"
FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

# ================= LOAD ARTIFACTS =================
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

with open(FEATURE_LIST_PATH, "r") as f:
    feature_data = json.load(f)

NUM_COLS = feature_data["num_cols"]
CAT_COLS = feature_data["cat_cols"]

# ================= LOAD DROPDOWN VALUES =================
train_df = pd.read_csv(TRAIN_CSV_PATH)

CAT_UNIQUES = {
    col: sorted(train_df[col].dropna().astype(str).unique().tolist())
    for col in CAT_COLS
}

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template(
        "index.html",
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        cat_uniques=CAT_UNIQUES
    )

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}

    for col in NUM_COLS:
        input_data[col] = float(request.form[col])

    for col in CAT_COLS:
        input_data[col] = request.form[col]

    df = pd.DataFrame([input_data])
    X_transformed = preprocessor.transform(df)
    prediction = model.predict(X_transformed)[0]

    return render_template(
        "index.html",
        prediction=round(float(prediction), 2),
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        cat_uniques=CAT_UNIQUES
    )

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    X_transformed = preprocessor.transform(df)
    preds = model.predict(X_transformed)

    df["Predicted_Price_INR"] = preds

    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="batch_predictions.csv"
    )

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
