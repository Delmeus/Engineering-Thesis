import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import torch
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from binaryModel.utils.ImageDataset import ImageDataset

random.seed(42)
np.random.seed(42)

clfs = [
    # sl.ensembles.SEA(GaussianNB(), n_estimators=3),
    # MLPClassifier(hidden_layer_sizes=(10), random_state=42),
    # sl.ensembles.AUE(base_estimator=GaussianNB()),
    # sl.ensembles.ROSE(base_estimator=GaussianNB()),
    sl.ensembles.OUSE(base_estimator=GaussianNB()),
    sl.ensembles.OOB(base_estimator=GaussianNB()),
    sl.ensembles.UOB(base_estimator=GaussianNB())
]

clf_names = [
    "OUSE",
    "OOB",
    "UOB"
    # "SEA",
    # # "MLP",
    # "AUE",
    # "ROSE"
]


def shuffle_stream(file_path):
    print("Shuffling dataset")
    dataset = np.load(file_path)
    np.random.shuffle(dataset)
    np.save("../npy_datasets/shuffeled_dataset.npy", dataset)

# dorobic kilka strumieni

stream = sl.streams.NPYParser('../npy_datasets/hog_dataset.npy', n_chunks=200, chunk_size=50)
stream.reset()

metrics = [sl.metrics.f1_score,
           sl.metrics.geometric_mean_score_1,
           sl.metrics.balanced_accuracy_score,
           sl.metrics.specificity]

metrics_names = ["F1 score",
                 "G-mean",
                 "Balanced accuracy score"]


evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream, clfs)

for m, metric in enumerate(metrics):
    plt.figure(figsize=(8, 6))

    plt.title(metrics_names[m])
    plt.ylim(0, 1)
    plt.xlabel("Chunk")
    plt.ylabel("Metric")

    for i, clf in enumerate(clfs):
        plt.plot(evaluator.scores[i, :, m], label=clf_names[i])

    plt.legend()

    plt.show()

from scipy.ndimage import gaussian_filter1d
for i, clf in enumerate(clfs):
    plt.figure(figsize=(8, 6))

    plt.title(clf_names[i])
    plt.ylim(0, 1)
    plt.xlabel("Chunk")
    plt.ylabel("Metric")

    for m, metric in enumerate(metrics):
        plt.plot(gaussian_filter1d(evaluator.scores[i, :, m], 1), label=metrics_names[m])
        print(f"{clf_names[i]}; srednia {metrics_names[m]} = {evaluator.scores[i, :, m].mean()}")

    plt.legend()

    plt.show()
