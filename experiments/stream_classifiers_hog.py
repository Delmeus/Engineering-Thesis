import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import torch
import random
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from binaryModel.utils.ImageDataset import ImageDataset
from scipy.ndimage import gaussian_filter1d

def plot_metrics_for_classifiers(evaluator, clfs, clf_names, metrics, metrics_names, stream):
    """
    Plots the metrics for each classifier along with the data stream.

    Args:
        evaluator: StreamLearn evaluator object containing the results.
        clfs: List of classifiers.
        clf_names: List of classifier names.
        metrics: List of metric functions.
        metrics_names: List of metric names.
        stream: Data stream object (used for plotting data characteristics, if needed).
    """
    num_chunks = evaluator.scores.shape[1]  # Number of chunks
    chunks = np.arange(num_chunks)  # Chunk indices

    for i, clf in enumerate(clfs):
        plt.figure(figsize=(12, 10))

        # Plot metrics
        for m, metric in enumerate(metrics):
            plt.subplot(4, 2, m + 1)  # Arrange in 4x2 grid
            metric_scores = evaluator.scores[i, :, m]
            smoothed_scores = gaussian_filter1d(metric_scores, 1)  # Smooth for better visualization

            plt.plot(chunks, smoothed_scores, label=metrics_names[m])
            plt.ylim(0, 1)
            plt.xlabel("Blok danych")
            plt.ylabel("Wartość metryki")
            plt.title(metrics_names[m])
            plt.grid(True)
            # plt.legend(loc="lower right")

        # Plot data stream (e.g., class distribution)
        plt.subplot(4, 2, 8)  # The last subplot
        plt.plot(imbalance)
        # Finalize the classifier's plot
        plt.tight_layout()
        plt.suptitle(f"Metrics for Classifier: {clf_names[i]}", fontsize=16)
        plt.subplots_adjust(top=0.9)  # Add some space for the main title
        plt.show()

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

metrics = [lambda y_true, y_pred: accuracy_score(y_true, y_pred),
           sl.metrics.recall,
           sl.metrics.precision,
           sl.metrics.specificity,
           sl.metrics.f1_score,
           sl.metrics.balanced_accuracy_score,
           sl.metrics.geometric_mean_score_1,
           ]

metrics_names = ["Dokładność",
                 "Czułość",
                 "Precyzja",
                 "Swoistość",
                 "F1",
                 "Zbalansowana dokładność",
                 "Średnia geometryczna"]


evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream, clfs)

imbalance = []
stream.reset()
# for i, (X_chunk, y_chunk) in enumerate(data_loader):
while not stream.is_dry():
    x_chunk, y_chunk = stream.get_chunk()
    y_chunk_np = y_chunk.numpy()
    sick_percentage = np.sum(y_chunk_np == 1) / len(y_chunk_np)
    imbalance.append(sick_percentage)

imbalance = np.array(imbalance) * 100

print(f"imbalance = {imbalance}")

plot_metrics_for_classifiers(evaluator, clfs, clf_names, metrics, metrics_names, imbalance)
# for m, metric in enumerate(metrics):
#     plt.figure(figsize=(8, 6))
#
#     plt.title(metrics_names[m])
#     plt.ylim(0, 1)
#     plt.xlabel("Chunk")
#     plt.ylabel("Metric")
#
#     for i, clf in enumerate(clfs):
#         plt.plot(evaluator.scores[i, :, m], label=clf_names[i])
#
#     plt.legend()
#
#     plt.show()
#
# from scipy.ndimage import gaussian_filter1d
# for i, clf in enumerate(clfs):
#     plt.figure(figsize=(8, 6))
#
#     plt.title(clf_names[i])
#     plt.ylim(0, 1)
#     plt.xlabel("Chunk")
#     plt.ylabel("Metric")
#
#     for m, metric in enumerate(metrics):
#         plt.plot(gaussian_filter1d(evaluator.scores[i, :, m], 1), label=metrics_names[m])
#         print(f"{clf_names[i]}; srednia {metrics_names[m]} = {evaluator.scores[i, :, m].mean()}")
#
#     plt.legend()
#
#     plt.show()
