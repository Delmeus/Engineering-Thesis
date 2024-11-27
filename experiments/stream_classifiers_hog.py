import numpy as np
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
# from skmultiflow.trees import HoeffdingTreeClassifier
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import sys


def plot_metrics_for_classifiers(evaluator, clfs, clf_names, metrics, metrics_names, filename: str):
    num_chunks = evaluator.scores.shape[1]
    chunks = np.arange(num_chunks)

    for i, clf in enumerate(clfs):
        plt.figure(figsize=(12, 10))

        for m, metric in enumerate(metrics):
            plt.subplot(4, 2, m + 1)
            metric_scores = evaluator.scores[i, :, m]
            smoothed_scores = gaussian_filter1d(metric_scores, 1)

            plt.plot(chunks, smoothed_scores, label=metrics_names[m])
            plt.ylim(0, 1)
            plt.xlim(0, 200)
            plt.xlabel("Blok danych")
            plt.ylabel("Wartość metryki")
            plt.title(metrics_names[m])
            plt.grid(True)

        plt.subplot(4, 2, 8)
        plt.plot(gaussian_filter1d(imbalance, 1), color='r')
        plt.xlim(0, 200)
        plt.grid(True)
        plt.xlabel("Blok danych")
        plt.ylabel("Stopień niezbalansowania [%]")
        plt.title("Strumień danych")
        plt.tight_layout()
        plt.savefig(f"../results/e2/eks2_{filename}_{clf_names[i]}.png", dpi=1200)
        plt.show()


if len(sys.argv) != 2:
    print("Invalid number of arguments")
    sys.exit(1)

if sys.argv[1] == "1":
    filename = "streams_pca_1.npy"
elif sys.argv[1] == "2":
    filename = "streams_pca_2.npy"
elif sys.argv[1] == "3":
    filename = "streams_pca_3.npy"
elif sys.argv[1] == "4":
    filename = "streams_pca_4.npy"
else:
    filename = "hog_dataset_with_pca.npy"

print(f"Chosen file {filename}")
random.seed(42)
np.random.seed(42)

clfs = [
    sl.ensembles.OUSE(base_estimator=GaussianNB()),
    sl.ensembles.OOB(base_estimator=GaussianNB()),
    sl.ensembles.UOB(base_estimator=GaussianNB())
]

clf_names = [
    "OUSE",
    "OOB",
    "UOB"
]


# def shuffle_stream(file_path):
#     print("Shuffling dataset")
#     dataset = np.load(file_path)
#     np.random.shuffle(dataset)
#     np.save("../npy_datasets/shuffeled_dataset.npy", dataset)


stream = sl.streams.NPYParser(f'../npy_datasets/{filename}', n_chunks=200, chunk_size=50)
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
num_chunks_processed = evaluator.scores.shape[1]


for i, clf_name in enumerate(clf_names):
    csv_output_file = f"../results/e2/{filename}_{clf_name}.csv"

    header = "Chunk," + ",".join(metrics_names)

    rows = []
    for chunk_idx in range(evaluator.scores.shape[1]):
        chunk_scores = evaluator.scores[i, chunk_idx, :]
        row = f"{chunk_idx}," + ",".join(map(str, chunk_scores))
        rows.append(row)

    # Write to the CSV file
    with open(csv_output_file, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows))

    print(f"Metrics per chunk saved for {clf_name} to {csv_output_file}")
imbalance = []
stream.reset()
while not stream.is_dry():
    x_chunk, y_chunk = stream.get_chunk()
    sick_percentage = np.sum(y_chunk == 1) / len(y_chunk)
    imbalance.append(sick_percentage)

imbalance = np.array(imbalance) * 100

print(f"imbalance = {imbalance}")

plot_metrics_for_classifiers(evaluator, clfs, clf_names, metrics, metrics_names, filename)

print(f"This experiment was run for file {filename}")
