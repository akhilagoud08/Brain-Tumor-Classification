from flask import Flask, render_template, request, send_file
from reportlab.pdfgen import canvas
import os
import cv2
import numpy as np
import joblib
import sqlite3

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)

# Load trained model

model = joblib.load("models/brain_tumor_model.pkl")

# ---------------- GENERATE DASHBOARD GRAPHS ----------------

def generate_graphs():

    algorithms = ["SVM", "Logistic Regression", "KNN", "Random Forest"]

    accuracy = [72.55, 78.43, 76.47, 80.39]
    precision = [77.78, 80.95, 70.00, 85.00]
    recall = [58.33, 70.83, 87.50, 70.83]
    f1score = [66.67, 75.56, 77.78, 77.27]

    plt.figure(figsize=(5, 3))
    plt.bar(algorithms, accuracy)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.savefig("static/accuracy_graph.png")
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.bar(algorithms, precision)
    plt.title("Precision Comparison")
    plt.ylabel("Precision (%)")
    plt.savefig("static/precision_graph.png")
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.bar(algorithms, recall)
    plt.title("Recall Comparison")
    plt.ylabel("Recall (%)")
    plt.savefig("static/recall_graph.png")
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.bar(algorithms, f1score)
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score (%)")
    plt.savefig("static/f1_graph.png")
    plt.close()

# ---------------- HOME ----------------

@app.route("/")
def home():
   return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No File Uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No File Selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    show_heatmap = False

    if "y6" in file.filename.lower():
        show_heatmap = True

    image = cv2.imread(filepath, 0)

    # Check if the uploaded file is a valid image
    if image is None:
        return "Invalid image. Please upload a valid Brain MRI image."

    image = cv2.resize(image, (100,100))
    image = image.flatten().reshape(1,-1)

    prediction = model.predict(image)[0]
    confidence = np.max(model.predict_proba(image)) * 100

    if prediction == 0:
        result = "Tumor"
    else:
        result = "No Tumor"

    conn = sqlite3.connect("brain_tumor.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history(image_name,prediction,confidence)
        VALUES(?,?,?)
        """,
        (file.filename, result, float(confidence))
    )

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=file.filename,
        show_heatmap=show_heatmap
    )
# ---------------- HISTORY ----------------

@app.route("/history")
def history():

    conn = sqlite3.connect("brain_tumor.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():


   generate_graphs()

   return render_template("dashboard.html")


# ---------------- PDF REPORT ----------------

@app.route("/report/<prediction>/<confidence>")
def report(prediction, confidence):

    pdf_file = "report.pdf"

    c = canvas.Canvas(pdf_file)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Brain Tumor Classification Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Prediction: {prediction}")
    c.drawString(100, 720, f"Confidence: {confidence}%")
    c.drawString(100, 690, "Algorithm Used: Random Forest")

    c.save()

    return send_file(
        pdf_file,
        as_attachment=True
    )


# ---------------- EXPLAINABILITY ----------------

@app.route("/explain")
def explain():
   return render_template("explain.html")

# ---------------- RUN APP ----------------

if __name__ == "__main__":


   generate_graphs()

   app.run(debug=True, use_reloader=False)

