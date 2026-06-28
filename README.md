# Brain Tumor Classification System

## Overview

This project is a Machine Learning based web application that classifies Brain MRI images as **Tumor** or **No Tumor**. The system uses a Random Forest classifier trained on MRI images and provides predictions through a Flask web application.

The project also includes prediction history storage, model comparison dashboard, PDF report generation, and an Explainable AI visualization module.

---

## Features

* MRI image upload and classification
* Brain Tumor / No Tumor prediction
* Prediction confidence score
* Compared SVM, Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest models
* Selected Random Forest as the final deployed classifier based on performance evaluation
* Accuracy, Precision, Recall, and F1-Score comparison dashboard
* Prediction history using SQLite database
* PDF report generation
* Explainability page with heatmap visualization
* Bootstrap-based responsive user interface
* Cloud deployment using Render

---

## Technologies Used

* Python
* Flask
* OpenCV
* Scikit-Learn
* SQLite
* Bootstrap
* ReportLab

---

## Algorithms Compared

* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Logistic Regression

Random Forest achieved the highest accuracy and was selected as the final prediction model.

---

## Project Structure

```text
Brain_Tumor_Classification/
│
├── dataset/
├── models/
│   └── brain_tumor_model.pkl
│
├── static/
│   ├── uploads/
│   ├── explanations/
│   ├── images/
│   └── css/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── dashboard.html
│   └── explain.html
│
├── app.py
├── train_model.py
├── create_database.py
├── compare_models.py
├── brain_tumor.db
└── README.md
```

---

## Dataset Structure

```text
dataset/
├── tumor/
└── notumor/
```

---

## How to Run

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python train_model.py

python create_database.py

python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## System Workflow

```text
MRI Image
      ↓
Image Preprocessing
      ↓
Feature Extraction
      ↓
Random Forest Model
      ↓
Prediction
      ↓
Result Display
```

---

## Explainable AI

The project includes an Explainability module demonstrating how important MRI regions can be visualized using heatmap-based explanations. Future versions can integrate advanced explainability techniques such as LIME and Grad-CAM.

---

## Future Enhancements

* Dynamic LIME Integration
* Grad-CAM Visualization
* Deep Learning (CNN) Models
* Multi-Class Tumor Classification
* Cloud Deployment

---

## Disclaismer

This project is developed for educational and research purposes only. It is not intended for medical diagnosis or clinical decision-making.
