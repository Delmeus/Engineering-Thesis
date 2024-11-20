file = open("G:/Projekty_Studia/inzynierka/results/e1/hog/imbalanced/SVM.csv", 'r')
data = file.read()
lines = data.strip().split("\n")

for line in lines:
    metric, *values = line.split(",")
    print(f"{metric.strip()}:")
    for value in values:
        print(f"  {value}")