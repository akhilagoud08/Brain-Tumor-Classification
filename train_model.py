import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = []
labels = []

categories = ['tumor', 'notumor']

for category in categories:
    path = os.path.join('dataset', category)

    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path, 0)

        image = cv2.resize(image, (100,100))

        data.append(image.flatten())
        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

joblib.dump(model,'models/brain_tumor_model.pkl')

print("Model Saved")