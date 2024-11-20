from binaryModel.utils.ResultHelper import print_mean_results

imb = False

if imb:
    path = "imbalanced"
else:
    path = "smote"

print("knn")
print_mean_results(f"results/e1/hog/{path}/kNN.csv")

print("\nNB")
print_mean_results(f"results/e1/hog/{path}/Naiwny klasyfikator Bayesowski.csv")

print("\nLR")
print_mean_results(f"results/e1/hog/{path}/Regresja logistyczna.csv")

print("\nsvm")
print_mean_results(f"results/e1/hog/{path}/SVM.csv")

