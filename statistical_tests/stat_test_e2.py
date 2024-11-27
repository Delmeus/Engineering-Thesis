import numpy as np
import pandas as pd
from statistical_Test import perform_test

metrics = ["Dokładność", "Czułość", "Precyzja", "Swoistość", "F1", "Zbalansowana dokładność", "Średnia geometryczna"]
methods = ["OOB", "UOB", "OUSE"]
filenames = ["hog_dataset_with_pca.npy", "streams_pca_1.npy", "streams_pca_2.npy", "streams_pca_3.npy", "streams_pca_4.npy"]

def metric_to_key(metric):
    if metric == "Dokładność":
        return 'Acc'
    elif metric == "Czułość":
        return 'Rec'
    elif metric == "Precyzja":
        return 'Prec'
    elif metric == "Swoistość":
        return 'Spec'
    elif metric == "F1":
        return 'F1'
    elif metric == "Zbalansowana dokładność":
        return 'BAC'
    elif metric == "Średnia geometryczna":
        return 'G-mean'
    else:
        print("Unknown metric")
        return ""

def get_values(path):
    file = open(path, 'r')
    data = file.read()
    rows = data.strip().split("\n")[1:]
    print(rows)
    results = {}
    for row in rows:
        columns = row.split(",")

        metric_name = columns[0]
        values = list(map(float, columns[1:]))
        results[metric_name] = values

    file.close()
    return results

oob_scores = {"Dokładność": [], "Czułość": [], "Precyzja": [], "Swoistość": [], "F1": [], "Zbalansowana dokładność": [], "Średnia geometryczna": []}
uob_scores = {"Dokładność": [], "Czułość": [], "Precyzja": [], "Swoistość": [], "F1": [], "Zbalansowana dokładność": [], "Średnia geometryczna": []}
ouse_scores = {"Dokładność": [], "Czułość": [], "Precyzja": [], "Swoistość": [], "F1": [], "Zbalansowana dokładność": [], "Średnia geometryczna": []}

for filename in filenames:
    oob = get_values(f'../results/e2/{filename}_OOB.csv')
    uob = get_values(f'../results/e2/{filename}_UOB.csv')
    ouse = get_values(f'../results/e2/{filename}_OUSE.csv')
    for metric in metrics:
        oob_scores[metric].extend(oob[metric])
        uob_scores[metric].extend(uob[metric])
        ouse_scores[metric].extend(ouse[metric])

results_df = pd.DataFrame(index=metrics, columns=methods)
results_df = results_df.map(lambda x: [])

print(results_df)

for metric in metrics:
    is_better = perform_test(oob_scores[metric], uob_scores[metric], metric, "OOB", "UOB")
    if is_better == 1:
        results_df.at[metric, "OOB"].append(2)
    elif is_better == 2:
        results_df.at[metric, "UOB"].append(1)

for metric in metrics:
    is_better = perform_test(oob_scores[metric], ouse_scores[metric], metric, "OOB", "OUSE")
    if is_better == 1:
        results_df.at[metric, "OOB"].append(3)
    elif is_better == 2:
        results_df.at[metric, "OUSE"].append(1)

for metric in metrics:
    is_better = perform_test(uob_scores[metric], ouse_scores[metric], metric, "UOB", "OUSE")
    if is_better == 1:
        results_df.at[metric, "UOB"].append(3)
    elif is_better == 2:
        results_df.at[metric, "OUSE"].append(2)

print(results_df)

results_array = np.array([
    [np.mean(oob_scores[metric]), np.mean(uob_scores[metric]), np.mean(ouse_scores[metric])]
    for metric in metrics
])

results_array = np.round(results_array, 3)

results_df = pd.DataFrame(
    results_array,
    columns=["OOB", "UOB", "OUSE"],
    index=metrics
)

print(results_df)
