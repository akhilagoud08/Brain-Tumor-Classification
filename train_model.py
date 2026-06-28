import cv2
import os
import numpy as np
import joblib

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
            print("Cannot read image:", img_path)
            continue

        image = cv2.resize(image, (100,100))

        data.append(image.flatten())

        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_accuracy = 0
best_model = None

print("\nMODEL PERFORMANCE\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred) * 100
    precision = precision_score(y_test, pred) * 100
    recall = recall_score(y_test, pred) * 100
    f1 = f1_score(y_test, pred) * 100

    print(f"\n{name}")
    print("-" * 30)
    print("Accuracy :", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall   :", round(recall, 2))
    print("F1 Score :", round(f1, 2))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

joblib.dump(
    best_model,
    'models/brain_tumor_model.pkl'
)

print("\nBest Model Saved Successfully")
print("Best Accuracy:", round(best_accuracy, 2))