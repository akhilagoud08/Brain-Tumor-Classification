import cv2
import os
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# Load Dataset
# -----------------------------

data = []
labels = []

categories = ['tumor', 'notumor']

for category in categories:

    path = os.path.join('dataset', category)

    label = categories.index(category)

    for img in os.listdir(path):

        img_path = os.path.join(path, img)

        image = cv2.imread(img_path, 0)

        if image is None:
            continue

        image = cv2.resize(image, (100, 100))

        data.append(image.flatten())
        labels.append(label)

# -----------------------------
# Convert to NumPy Arrays
# -----------------------------

X = np.array(data)
y = np.array(labels)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------

models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# -----------------------------
# Evaluation
# -----------------------------

print("\n" + "=" * 50)
print("MODEL COMPARISON RESULTS")
print("=" * 50)

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"\n{name}")
    print("-" * 30)

    print("Accuracy  :", round(accuracy * 100, 2), "%")
    print("Precision :", round(precision * 100, 2), "%")
    print("Recall    :", round(recall * 100, 2), "%")
    print("F1 Score  :", round(f1 * 100, 2), "%")

print("\nComparison Completed Successfully!")