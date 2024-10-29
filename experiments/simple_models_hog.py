import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os
from binaryModel.utils.ResultHelper import ResultHelper
from sklearn.model_selection import RepeatedStratifiedKFold



def load_images_from_csv(csv_file, img_folder):
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for index, row in data.iterrows():
        img_path = os.path.join(img_folder, row['image']) + '.jpg'
        img_path = os.path.normpath(img_path)
        label = row['sick']

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (64, 128))
            images.append(img)
            labels.append(label)
        else:
            print("Can't read image!")

    return images, labels


csv_file = '../binaryModel/archive/labels.csv'
img_folder = '../img/archive/all_images'
images, labels = load_images_from_csv(csv_file, img_folder)


hog = cv2.HOGDescriptor()


def extract_hog_features(images):
    features = []
    for img in images:
        hog_features = hog.compute(img).flatten()
        features.append(hog_features)
    return np.array(features)


X = extract_hog_features(images)

y = np.array(labels)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=36851234)

svm_results = ResultHelper("SVM_hog")
kNN_results = ResultHelper("kNN_hog")
nB_results = ResultHelper("Naive Bayes_hog")
logistic_results = ResultHelper("Logistic regression_hog")

iteration = 1

for train_index, test_index in rskf.split(X, y):
    print(f"iteration = {iteration}")
    iteration = iteration + 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Training SVM")
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    svm_results.append_all_scores(y_test, y_pred)

    print("Training kNN")
    kNN_classifier = KNeighborsClassifier(n_neighbors=3)
    kNN_classifier.fit(X_train, y_train)
    y_pred = kNN_classifier.predict(X_test)
    kNN_results.append_all_scores(y_test, y_pred)

    print("Training NB")
    nB_classifier = GaussianNB()
    nB_classifier.fit(X_train, y_train)
    y_pred = nB_classifier.predict(X_test)
    nB_results.append_all_scores(y_test, y_pred)

    print("Training logistic regression")
    logistic_classifier = LogisticRegression(max_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    y_pred = logistic_classifier.predict(X_test)
    logistic_results.append_all_scores(y_test, y_pred)


svm_results.plot_radar_chart_for_ml_models()
svm_results.save_scores()

kNN_results.plot_radar_chart_for_ml_models()
kNN_results.save_scores()

nB_results.plot_radar_chart_for_ml_models()
nB_results.save_scores()

logistic_results.plot_radar_chart_for_ml_models()
logistic_results.save_scores()


models = ({
    "SVM": svm_results.scores,
    "kNN": kNN_results.scores,
    "Naive Bayes": nB_results.scores,
    "Logistic Regression": logistic_results.scores
    })

ResultHelper.plot_radar_combined(models)