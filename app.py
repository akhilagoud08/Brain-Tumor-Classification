import json
import pickle
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from skimage.feature import hog, local_binary_pattern
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.secret_key = "replace-this-secret-key"
UPLOAD_DIR.mkdir(exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_artifacts():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def preprocess_image(image_path, image_size=(200, 200)):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    image = cv2.resize(image, tuple(image_size))
    return image.astype("float32") / 255.0


def extract_features(image, method):
    if method == "raw":
        return image.flatten()
    if method == "hog":
        return hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
    if method == "lbp":
        radius = 2
        points = 8 * radius
        lbp = local_binary_pattern(image, points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
        hist = hist.astype("float32")
        return hist / (hist.sum() + 1e-7)
    raise ValueError(f"Unsupported feature method: {method}")


def read_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    artifact = load_artifacts()
    if artifact is None:
        flash("Model not found. Run python train_model.py after adding the Kaggle dataset.")
        return redirect(url_for("predict"))

    file = request.files.get("mri_image")
    if file is None or file.filename == "":
        flash("Please upload an MRI image.")
        return redirect(url_for("predict"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Upload PNG, JPG, JPEG, BMP, TIF, or TIFF.")
        return redirect(url_for("predict"))

    filename = f"{uuid4().hex}_{secure_filename(file.filename)}"
    saved_path = UPLOAD_DIR / filename
    file.save(saved_path)

    try:
        image = preprocess_image(saved_path, artifact.get("image_size", (200, 200)))
        features = extract_features(image, artifact["feature_method"]).reshape(1, -1)
        features = artifact["scaler"].transform(features)
        model = artifact["model"]
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities) * 100)
        probability_map = {label: float(prob * 100) for label, prob in zip(model.classes_, probabilities)}
    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("predict"))

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probability_map,
        image_url=url_for("uploaded_file", filename=filename),
        feature_method=artifact["feature_method"].upper(),
        model_name=artifact["metrics"]["model_name"],
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/dashboard")
def dashboard():
    metrics = read_metrics()
    return render_template("dashboard.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
