from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os


class ResultHelper:
    def __init__(self, name="model"):
        self.name = name
        self.metrics = ["accuracy", "recall", "precision", "f1", "balanced accuracy"]
        self.scores = {"accuracy": 0.0,
                       "recall": 0.0,
                       "precision": 0.0,
                       "f1": 0.0,
                       "balanced accuracy": 0.0}
        # self.accuracy_scores = []
        # self.recall_scores = []
        # self.precision_scores = []
        # self.f1_scores = []
        # self.balanced_accuracy_scores = []

    def append_all_scores(self, y_test, y_pred):
        # self.accuracy_scores.append(round(accuracy_score(y_test, y_pred), 3))
        # self.recall_scores.append(round(recall_score(y_test, y_pred), 3))
        # self.precision_scores.append(round(precision_score(y_test, y_pred), 3))
        # self.f1_scores.append(round(f1_score(y_test, y_pred), 3))
        # self.balanced_accuracy_scores.append(round(balanced_accuracy_score(y_test, y_pred), 3))

        self.scores[self.metrics[0]] = round(accuracy_score(y_test, y_pred), 3)
        self.scores[self.metrics[1]] = round(recall_score(y_test, y_pred), 3)
        self.scores[self.metrics[2]] = round(precision_score(y_test, y_pred), 3)
        self.scores[self.metrics[3]] = round(f1_score(y_test, y_pred), 3)
        self.scores[self.metrics[4]] = round(balanced_accuracy_score(y_test, y_pred), 3)

    def print_all_scores(self):
        for metric in self.metrics:
            print(f"{metric} = {self.scores[metric]}")
        # print(f"accuracy scores = {self.accuracy_scores}")
        # print(f"recall scores = {self.recall_scores}")
        # print(f"precision scores = {self.precision_scores}")
        # print(f"f1 scores = {self.f1_scores}")
        # print(f"balanced accuracy scores = {self.f1_scores}")

    def print_mean_scores(self):
        for metric in self.metrics:
            print(f"{metric} = {round(np.mean(self.scores[metric]), 3)}")
        # print(f"Mean values from {len(self.accuracy_scores)} experiments")
        # print(f"accuracy scores = {round(np.mean(self.accuracy_scores), 2)}")
        # print(f"recall scores = {round(np.mean(self.recall_scores), 2)}")
        # print(f"precision scores = {round(np.mean(self.precision_scores), 2)}")
        # print(f"f1 scores = {round(np.mean(self.f1_scores), 2)}")
        # print(f"balanced accuracy scores = {round(np.mean(self.f1_scores), 2)}")

    def save_scores(self):
        path = f"./results/{self.name}.csv"
        path = os.path.normpath(path)
        file = open(path, "w")
        for metric in self.metrics:
            file.write(f"{metric},{round(np.mean(self.scores[metric]), 3)}\n")
        file.close()

    def plot_radar_chart(self):
        means = []
        for metric in self.metrics:
            means.append(self.scores[metric])

        N = 5
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], self.metrics)

        ax.set_rlabel_position(0)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                   color="grey", size=7)
        plt.ylim(0,1)

        means += means[:1]
        ax.plot(angles, means, linewidth=1, linestyle='solid')

        plt.title(f"{self.name} Performance Chart", size=20, color='black', y=1.06)
        path = f"./results/radar_{self.name}.png"
        path = os.path.normpath(path)
        plt.savefig(path, dpi=200)
        plt.show()

    @staticmethod
    def plot_radar_combined(models: dict):
        metrics = ["accuracy", "recall", "precision", "f1", "balanced accuracy"]

        N = 5
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], metrics)

        ax.set_rlabel_position(0)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                   color="grey", size=7)
        plt.ylim(0,1)

        for name, scores in models.items():
            values = []
            for metric in metrics:
                values.append(scores[metric])
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=name)

        plt.title("Combined Performance Chart", size=20, color='black', y=1.06)
        path = "./results/radar_combined.png"
        path = os.path.normpath(path)
        plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
        plt.savefig(path, dpi=200)
        plt.show()
