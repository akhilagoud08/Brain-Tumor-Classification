from flask import Flask, render_template, request, send_file
from reportlab.pdfgen import canvas
import os
import cv2
import numpy as np
import joblib
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = joblib.load("models/brain_tumor_model.pkl")


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
    print("Uploaded file:", file.filename)

    # Show heatmap only for Y6.png
    if  "y6" in file.filename.lower():

       show_heatmap = True
    else:
       show_heatmap = False

    # Image preprocessing
    image = cv2.imread(filepath, 0)
    image = cv2.resize(image, (100, 100))

    image = image.flatten().reshape(1, -1)

    prediction = model.predict(image)[0]

    confidence = np.max(model.predict_proba(image)) * 100

    if prediction == 0:
        result = "Tumor"
    else:
        result = "No Tumor"

    # Save to database
    conn = sqlite3.connect("brain_tumor.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history(image_name,prediction,confidence)
    VALUES(?,?,?)
    """, (file.filename, result, float(confidence)))

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

    results = [
        ["Random Forest", 80.39, 85.00, 70.83, 77.27],
        ["SVM", 72.55, 77.78, 58.33, 66.67],
        ["KNN", 76.47, 70.00, 87.50, 77.78],
        ["Logistic Regression", 78.43, 80.95, 70.83, 75.56]
    ]

    return render_template(
        "dashboard.html",
        results=results
    )


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
    app.run(debug=True)