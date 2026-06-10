# Brain Tumor Classification Web Application

This project is a Flask-based Brain MRI classification system using traditional machine learning. It preprocesses MRI images, extracts Raw, HOG, and LBP features, trains six classifiers, selects the best-performing model, and serves predictions through a responsive medical-themed web UI.

## Features

- Grayscale conversion, 200x200 resizing, and pixel normalization
- Raw image, Histogram of Oriented Gradients, and Local Binary Pattern features
- SVM, Logistic Regression, KNN, Naive Bayes, Decision Tree, and Random Forest comparison
- 80% training and 20% testing split
- Accuracy, precision, sensitivity, specificity, ROC AUC, confusion matrix, and classification report
- Automatic best model selection and Pickle model saving
- Flask pages for home, about, upload, result, and dashboard
- Uploaded MRI image preview and stored uploads

## Project Structure

```text
Brain_Tumor_Classification/
├── dataset/
├── models/
├── uploads/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset Setup

Download a Brain MRI dataset from Kaggle and place it in `dataset/` with two class folders. Supported folder names include:

```text
dataset/
├── yes/
└── no/
```

or:

```text
dataset/
├── Tumor/
└── No Tumor/
```

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Render Deployment

1. Push this project to GitHub.
2. Create a new Render Web Service from the repository.
3. Set the build command:

```bash
pip install -r requirements.txt
```

4. Set the start command:

```bash
gunicorn app:app
```

5. Upload or include a trained `models/best_model.pkl` and `models/metrics.json` before deployment. Render instances have ephemeral storage, so train locally and commit or attach the model artifacts for demo use.

## Notes

This application is for education and research demonstration only. It is not a medical diagnostic system.
