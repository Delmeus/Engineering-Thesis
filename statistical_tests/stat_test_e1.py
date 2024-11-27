from statistical_Test import perform_test
from binaryModel.utils.ResultHelper import get_values
from binaryModel.utils.ResultHelper import ResultHelper
from tabulate import tabulate
from numpy import mean

dataset = "imbalanced"
rh = ResultHelper()
metrics = ['Acc', 'Rec', 'Prec', 'Spec', 'F1', 'BAC', 'G-mean']
knn = get_values(f"../results/e1/hog/{dataset}/kNN.csv")
nb = get_values(f"../results/e1/hog/{dataset}/Naiwny klasyfikator Bayesowski.csv")
svm = get_values(f"../results/e1/hog/{dataset}/SVM.csv")
rl = get_values(f"../results/e1/hog/{dataset}/Regresja logistyczna.csv")

models = {
    "kNN": knn,
    "NB": nb,
    "RL": rl,
    "SVC": svm,
}

rh.plot_radar_combined(models, "./")



def gather_and_print_metrics(metrics):
    """
    Collects metrics from all models, computes mean values,
    and prints a formatted table of results.
    """
    # Prepare the table header
    header = ["Metric", "Model", "Mean Value"]

    # Initialize rows for the table
    rows = []

    # Loop through each metric
    for metric in metrics:
        for model_name, scores in models.items():
            mean_value = mean(scores[metric])
            rows.append([metric, model_name, f"{mean_value:.3f}"])

    # Print the table using tabulate
    print(tabulate(rows, headers=header, tablefmt="grid"))

# Call the function with the example metrics
gather_and_print_metrics(metrics)

def translate_name(name):
    if name == "kNN":
        return 1
    elif name == "NB":
        return 2
    elif name == "RL":
        return 3
    elif name == "SVM":
        return 4
    return -1

for i, (model_a_name, model_a_scores) in enumerate(models.items()):
    for j, (model_b_name, model_b_scores) in enumerate(models.items()):
        if i < j:
            print(f"\nComparing {model_a_name} and {model_b_name}")
            for metric in metrics:
                perform_test(model_a_scores[metric], model_b_scores[metric], metric, translate_name(model_a_name), translate_name(model_b_name))

# knnSMOTE = get_values(f"../results/e1/hog/SMOTE/kNN.csv")
# nbSMOTE = get_values(f"../results/e1/hog/SMOTE/Naiwny klasyfikator Bayesowski.csv")
# svmSMOTE = get_values(f"../results/e1/hog/SMOTE/SVM.csv")
# rlSMOTE = get_values(f"../results/e1/hog/SMOTE/Regresja logistyczna.csv")
#
# smote = {
#     "kNN": knnSMOTE,
#     "NB": nbSMOTE,
#     "RL": rlSMOTE,
#     "SVC": svmSMOTE,
# }
#
# for model_name, model_scores in smote.items():
#     print(f"\nComparing {model_name}")
#     for metric in metrics:
#         perform_test(model_scores[metric], models[model_name][metric], metric, "smote", "imb")