import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import os
from binaryModel.utils import ResultHelper as RH
import joblib

from sklearn.model_selection import RepeatedStratifiedKFold

data = np.load('../npy_datasets/image_dataset.npy')

X = data[:, :-1]
y = data[:, -1]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=36851234)

svm_results = RH.ResultHelper("SVM")
kNN_results = RH.ResultHelper("kNN")
nB_results = RH.ResultHelper("Naive Bayes")
logistic_results = RH.ResultHelper("Logistic regression")
# adaBoost_results = RH.ResultHelper("AdaBoost")

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
    #
    # print("Training AdaBoost")
    # adaBoost_classifier = AdaBoostClassifier(n_estimators=50)
    # adaBoost_classifier.fit(X_train, y_train)
    # y_pred = adaBoost_classifier.predict(X_test)
    # adaBoost_results.append_all_scores(y_test, y_pred)

# print("\nSVM results:")
# svm_results.print_all_scores()
# svm_results.print_mean_scores()
svm_results.plot_radar_chart_for_ml_models()
svm_results.save_scores()

# print("\nkNN results:")
# kNN_results.print_all_scores()
# kNN_results.print_mean_scores()
kNN_results.plot_radar_chart_for_ml_models()
kNN_results.save_scores()

nB_results.plot_radar_chart_for_ml_models()
nB_results.save_scores()

logistic_results.plot_radar_chart_for_ml_models()
logistic_results.save_scores()


# print("\nAdaBoost results:")
# adaBoost_results.print_all_scores()
# adaBoost_results.print_mean_scores()
# adaBoost_results.plot_radar_chart_for_ml_models()
# adaBoost_results.save_scores()

models = ({
    "SVM": svm_results.scores,
    "kNN": kNN_results.scores,
    "Naive Bayes": nB_results.scores,
    "Logistic Regression": logistic_results.scores
    # "AdaBoost": adaBoost_results.scores
    })

RH.ResultHelper.plot_radar_combined(models)

# def predict_single_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (64, 128))
#     hog_features = hog.compute(img).flatten()
#     prediction = svm_classifier.predict([hog_features])
#     return prediction[0]


# img_path = 'test.png'
# predicted_label = predict_single_image(img_path)
# print(f"Predicted label for the test image: {predicted_label}")
#
# joblib.dump(svm_classifier, 'svm_classifier_model.joblib')