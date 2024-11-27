import numpy as np
import pandas as pd
from statistical_Test import perform_test

untrained_scores = {'acc': [], 'prec': [], 'rec': [], 'spec': [], 'f1': [], 'bac': [], 'gmean': []}
pretrained_scores = {'acc': [], 'prec': [], 'rec': [], 'spec': [], 'f1': [], 'bac': [], 'gmean': []}
metrics = ['acc', 'rec', 'prec', 'spec', 'f1', 'bac', 'gmean']
methods = ["My network", "Resnet18"]
def get_values(path):
    file = open(path, 'r')
    data = file.read()
    rows = data.strip().split("\n")
    results = {}
    for row in rows:
        columns = row.split(",")
        metric_name = columns[0]
        values = []
        values.append(float(columns[-1]))
        results[metric_name] = values

    file.close()
    return results


for i in range(5):
    untrained = get_values(f"../results/e3/test_run_{i}.csv")
    trained = get_values(f"../results/e3/pretrained/test_run_{i}.csv")
    for metric in metrics:
        untrained_scores[metric].extend(untrained[metric])
        pretrained_scores[metric].extend(trained[metric])

print(untrained_scores)
results_array = np.zeros((7, 2))

results_df = pd.DataFrame(
    results_array,
    index=metrics,
    columns=methods
)


for metric in metrics:
    is_better = perform_test(untrained_scores[metric], pretrained_scores[metric], metric, "My network", "RESNET18")
    if is_better == 1:
        results_df.at[metric, "My network"] = 2
    elif is_better == 2:
        results_df.at[metric, "Resnet18"] = 1

print(results_df)

results_array = np.array([
    [np.mean(untrained_scores[metric]), np.mean(pretrained_scores[metric])]
    for metric in metrics
])

results_array = np.round(results_array, 3)

results_df = pd.DataFrame(
    results_array,
    columns=["Untrained", "Resnet18"],
    index=metrics
)

print(results_df)