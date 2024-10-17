import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import os
import joblib


# 1. Load image paths and labels from CSV file
def load_images_from_csv(csv_file, img_folder):
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for index, row in data.iterrows():
        img_path = os.path.join(img_folder, row['image']) + '.jpg'
        label = row['sick']

        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (64, 128))  # Resize to a fixed size for HOG
            images.append(img)
            labels.append(label)
        else:
            print("Can't read image!")

    return images, labels


# Load the images and labels from the CSV file
csv_file = 'img/archive/labels.csv'  # Path to your label CSV file
img_folder = 'img/archive/all_images'  # Folder where your images are stored
images, labels = load_images_from_csv(csv_file, img_folder)

# 2. Extract HOG features
hog = cv2.HOGDescriptor()


def extract_hog_features(images):
    features = []
    for img in images:
        hog_features = hog.compute(img).flatten()  # Compute HOG and flatten to a vector
        features.append(hog_features)
    return np.array(features)


# Extract HOG features for all images
X = extract_hog_features(images)

# Convert labels to a NumPy array
y = np.array(labels)

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train an SVM model for binary classification (sick vs not_sick)
svm_classifier = SVC(kernel='linear')  # Linear kernel for simplicity
svm_classifier.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
recall_score = recall_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\nTest F1: {f1_score * 100:.2f}%\nTest recall score: {recall_score * 100:.2f}%")


# Function to predict a single image using the trained SVM model
def predict_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 128))  # Resize to match training data
    hog_features = hog.compute(img).flatten()
    prediction = svm_classifier.predict([hog_features])
    return prediction[0]  # Return predicted label (0 or 1)


# Test with a new image
img_path = 'test.png'
predicted_label = predict_single_image(img_path)
print(f"Predicted label for the test image: {predicted_label}")

joblib.dump(svm_classifier, 'svm_classifier_model.joblib')