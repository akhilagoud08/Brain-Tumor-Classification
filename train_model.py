import json
import os
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
STATIC_IMAGE_DIR = BASE_DIR / "static" / "images"
IMAGE_SIZE = (200, 200)
RANDOM_STATE = 42
POSITIVE_HINTS = ("yes", "tumor", "positive", "brain_tumor")
NEGATIVE_HINTS = ("no", "notumor", "no_tumor", "negative", "normal", "healthy")


def normalize_label(folder_name):
    name = folder_name.lower().replace(" ", "_").replace("-", "_")
    if any(hint in name for hint in NEGATIVE_HINTS):
        return "No Tumor"
    if any(hint in name for hint in POSITIVE_HINTS):
        return "Tumor"
    raise ValueError(
        f"Could not infer label from folder '{folder_name}'. Use folders like yes/no, Tumor/No Tumor, or Normal/Tumor."
    )


def preprocess_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, IMAGE_SIZE)
    return image.astype("float32") / 255.0


def raw_features(image):
    return image.flatten()


def hog_features(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def lbp_features(image):
    radius = 2
    points = 8 * radius
    lbp = local_binary_pattern(image, points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype("float32")
    return hist / (hist.sum() + 1e-7)


FEATURE_EXTRACTORS = {
    "raw": raw_features,
    "hog": hog_features,
    "lbp": lbp_features,
}


def load_dataset():
    images, labels = [], []
    if not DATASET_DIR.exists():
        raise FileNotFoundError("dataset folder not found. Place Kaggle MRI folders inside dataset/ before training.")

    image_paths = []
    for folder in DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue
        label = normalize_label(folder.name)
        for path in folder.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                image_paths.append((path, label))

    if not image_paths:
        raise FileNotFoundError("No image files found. Expected dataset/<class_name>/*.jpg or similar.")

    for path, label in image_paths:
        try:
            images.append(preprocess_image(path))
            labels.append(label)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")

    if len(set(labels)) != 2:
        raise ValueError(f"Expected two classes after label normalization, found: {sorted(set(labels))}")

    return np.array(images), np.array(labels)


def specificity_score(y_true, y_pred, negative_label="No Tumor"):
    labels = ["No Tumor", "Tumor"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def get_models():
    return {
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    }


def extract_matrix(images, method):
    extractor = FEATURE_EXTRACTORS[method]
    return np.array([extractor(image) for image in images])


def predict_scores(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, list(model.classes_).index("Tumor")]
    decision = model.decision_function(x_test)
    return (decision - decision.min()) / (decision.max() - decision.min() + 1e-7)


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=["No Tumor", "Tumor"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Tumor", "Tumor"], yticklabels=["No Tumor", "Tumor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_score, output_path):
    y_binary = (y_true == "Tumor").astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_score)
    auc = roc_auc_score(y_binary, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#8a98a8")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    STATIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    images, labels = load_dataset()
    x_train_images, x_test_images, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    results = []
    best = None

    for feature_name in FEATURE_EXTRACTORS:
        x_train = extract_matrix(x_train_images, feature_name)
        x_test = extract_matrix(x_test_images, feature_name)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for model_name, model in get_models().items():
            print(f"Training {model_name} with {feature_name.upper()} features...")
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            y_score = predict_scores(model, x_test_scaled)

            metrics = {
                "feature_method": feature_name,
                "model_name": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label="Tumor", zero_division=0),
                "sensitivity_recall": recall_score(y_test, y_pred, pos_label="Tumor", zero_division=0),
                "specificity": specificity_score(y_test, y_pred),
                "roc_auc": roc_auc_score((y_test == "Tumor").astype(int), y_score),
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            }
            results.append(metrics)

            candidate = {
                "model": model,
                "scaler": scaler,
                "feature_method": feature_name,
                "metrics": metrics,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_score": y_score,
            }
            if best is None or metrics["accuracy"] > best["metrics"]["accuracy"]:
                best = candidate

    with open(MODEL_DIR / "best_model.pkl", "wb") as file:
        pickle.dump(
            {
                "model": best["model"],
                "scaler": best["scaler"],
                "feature_method": best["feature_method"],
                "image_size": IMAGE_SIZE,
                "classes": ["No Tumor", "Tumor"],
                "metrics": best["metrics"],
            },
            file,
        )

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"all_results": results, "best_model": best["metrics"]}, file, indent=2)

    plot_confusion_matrix(best["y_test"], best["y_pred"], STATIC_IMAGE_DIR / "confusion_matrix.png")
    plot_roc_curve(best["y_test"], best["y_score"], STATIC_IMAGE_DIR / "roc_curve.png")
    print(f"Best model: {best['metrics']['model_name']} using {best['feature_method'].upper()} features")
    print(f"Accuracy: {best['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    train()
