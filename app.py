# # from flask import Flask, render_template, request, send_file
# # import pandas as pd
# # import joblib
# # import json
# # from pathlib import Path
# # import io
# # import webbrowser
# # import threading

# # # ================= APP INIT =================
# # app = Flask(__name__)

# # # ================= PATHS =================
# # PROJECT_ROOT = Path(__file__).parent

# # PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
# # MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "models" / "current_model.joblib"
# # FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
# # TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

# # # ================= LOAD ARTIFACTS =================
# # preprocessor = joblib.load(PREPROCESSOR_PATH)
# # model = joblib.load(MODEL_PATH)

# # with open(FEATURE_LIST_PATH, "r") as f:
# #     feature_data = json.load(f)

# # NUM_COLS = feature_data["num_cols"]
# # CAT_COLS = feature_data["cat_cols"]
# # FEATURES = NUM_COLS + CAT_COLS

# # # ================= LOAD DROPDOWN VALUES =================
# # train_df = pd.read_csv(TRAIN_CSV_PATH)

# # CAT_UNIQUES = {
# #     col: sorted(train_df[col].dropna().astype(str).unique().tolist())
# #     for col in CAT_COLS
# # }

# # # ================= ROUTES =================
# # @app.route("/")
# # def home():
# #     return render_template(
# #         "index.html",
# #         num_cols=NUM_COLS,
# #         cat_cols=CAT_COLS,
# #         cat_uniques=CAT_UNIQUES
# #     )


# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     input_data = {}

# #     # numeric inputs
# #     for col in NUM_COLS:
# #         input_data[col] = float(request.form[col])

# #     # categorical inputs
# #     for col in CAT_COLS:
# #         input_data[col] = request.form[col]

# #     df = pd.DataFrame([input_data])

# #     X_transformed = preprocessor.transform(df)
# #     prediction = model.predict(X_transformed)[0]

# #     return render_template(
# #         "index.html",
# #         prediction=round(float(prediction), 2),
# #         num_cols=NUM_COLS,
# #         cat_cols=CAT_COLS,
# #         cat_uniques=CAT_UNIQUES
# #     )


# # @app.route("/batch_predict", methods=["POST"])
# # def batch_predict():
# #     file = request.files["file"]
# #     df = pd.read_csv(file)

# #     X_transformed = preprocessor.transform(df)
# #     preds = model.predict(X_transformed)

# #     df["Predicted_Price_INR"] = preds

# #     output = io.BytesIO()
# #     df.to_csv(output, index=False)
# #     output.seek(0)

# #     return send_file(
# #         output,
# #         mimetype="text/csv",
# #         as_attachment=True,
# #         download_name="batch_predictions.csv"
# #     )

# # # ================= AUTO OPEN BROWSER =================
# # def open_browser():
# #     webbrowser.open_new("http://127.0.0.1:5000")

# # # ================= MAIN =================
# # if __name__ == "__main__":
# #     threading.Timer(1.0, open_browser).start()
# #     app.run(debug=True)










# from flask import Flask, render_template, request, send_file, jsonify
# import pandas as pd
# import joblib
# import json
# from pathlib import Path
# import io
# import os

# app = Flask(__name__)

# # =====================================================
# # PATH CONFIG
# # =====================================================
# PROJECT_ROOT = Path(__file__).parent

# MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "models" / "current_model.joblib"
# PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
# FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
# TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"

# # =====================================================
# # LOAD ARTIFACTS (ON STARTUP)
# # =====================================================
# preprocessor = joblib.load(PREPROCESSOR_PATH)
# model = joblib.load(MODEL_PATH)

# with open(FEATURE_LIST_PATH) as f:
#     feature_data = json.load(f)

# NUM_COLS = feature_data["num_cols"]
# CAT_COLS = feature_data["cat_cols"]

# # Dropdown values (from training data)
# train_df = pd.read_csv(TRAIN_CSV_PATH)
# CAT_UNIQUES = {
#     col: sorted(train_df[col].dropna().astype(str).unique().tolist())
#     for col in CAT_COLS
# }

# # =====================================================
# # GLOBAL STORE (for batch download)
# # =====================================================
# batch_result_df = None

# # =====================================================
# # ROUTES
# # =====================================================

# @app.route("/")
# def home():
#     return render_template(
#         "index.html",
#         num_cols=NUM_COLS,
#         cat_cols=CAT_COLS,
#         cat_uniques=CAT_UNIQUES
#     )


# # ---------------- SINGLE PREDICTION ------------------
# @app.route("/predict", methods=["POST"])
# def predict():
#     input_data = {}

#     for col in NUM_COLS:
#         input_data[col] = float(request.form[col])

#     for col in CAT_COLS:
#         input_data[col] = request.form[col]

#     df = pd.DataFrame([input_data])
#     X_transformed = preprocessor.transform(df)
#     prediction = model.predict(X_transformed)[0]

#     return render_template(
#         "index.html",
#         prediction=round(prediction, 2),
#         num_cols=NUM_COLS,
#         cat_cols=CAT_COLS,
#         cat_uniques=CAT_UNIQUES
#     )


# # ---------------- BATCH PREDICTION + PREVIEW ----------
# @app.route("/batch_predict", methods=["POST"])
# def batch_predict():
#     global batch_result_df

#     file = request.files["file"]
#     df = pd.read_csv(file)

#     X_transformed = preprocessor.transform(df)
#     preds = model.predict(X_transformed)

#     df["Predicted_Price_INR"] = preds.round(2)
#     batch_result_df = df.copy()

#     preview_df = df.head(10)

#     return render_template(
#         "batch_preview.html",
#         tables=[preview_df.to_html(classes="table", index=False)]
#     )


# # ---------------- DOWNLOAD FULL CSV ------------------
# @app.route("/download_batch")
# def download_batch():
#     global batch_result_df

#     if batch_result_df is None:
#         return "No batch prediction available", 400

#     output = io.BytesIO()
#     batch_result_df.to_csv(output, index=False)
#     output.seek(0)

#     return send_file(
#         output,
#         mimetype="text/csv",
#         as_attachment=True,
#         download_name="batch_predictions.csv"
#     )


# # ---------------- HEALTH CHECK -----------------------
# @app.route("/health")
# def health():
#     return jsonify({
#         "status": "UP",
#         "model_loaded": True,
#         "service": "Laptop Price Predictor"
#     })


# # =====================================================
# # ENTRY POINT (RENDER SAFE)
# # =====================================================
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)













from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import json
from pathlib import Path
import io
import webbrowser
import threading
import os

# ================= APP INIT =================
app = Flask(__name__)

# ================= PATHS =================
PROJECT_ROOT = Path(__file__).parent

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
FEATURES = NUM_COLS + CAT_COLS

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

    # numeric inputs
    for col in NUM_COLS:
        input_data[col] = float(request.form[col])

    # categorical inputs
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

# ================= AUTO OPEN BROWSER (SAFE) =================
def open_browser():
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open_new("http://127.0.0.1:5000")

# ================= MAIN =================
if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True, use_reloader=False)
