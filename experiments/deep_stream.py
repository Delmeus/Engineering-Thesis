import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
)
from imblearn.metrics import geometric_mean_score, specificity_score
from binaryModel.utils.ImageDataset import ImageDataset
from scipy.ndimage import gaussian_filter1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
score_names = ["F1", "Czułość", "Precyzja", "Dokładność", "Zbalansowana dokładność", "Średnia geometryczna", "Swoistość"]


def plot_batch_scores(batch_scores, data_loader, batch_number):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))
    axes = axes.ravel()
    metric_keys = ["f1", "rec", "prec", "acc", "bac", "gmean", "spec"]

    for i, metric in enumerate(metric_keys):
        axes[i].plot(gaussian_filter1d(batch_scores[metric], 2), marker='None', label=metric)
        axes[i].set_title(score_names[i])
        axes[i].set_xlabel('Blok danych')
        axes[i].set_ylabel('Średni wynik')
        axes[i].grid(True)

    imbalance_levels = []
    for i, (X_chunk, y_chunk) in enumerate(data_loader):
        y_chunk_np = y_chunk.numpy()
        sick_percentage = np.sum(y_chunk_np == 1) / len(y_chunk_np)
        imbalance_levels.append(sick_percentage)

    imbalance_levels = np.array(imbalance_levels) * 100
    axes[7].plot(gaussian_filter1d(imbalance_levels, 1), color='r')
    axes[7].set_title("Strumień danych")
    axes[7].set_xlabel('Blok danych')
    axes[7].set_ylabel('Stopień niezbalansowania')
    axes[7].grid(True)
    # for j in range(len(metric_keys), len(axes)):
    #     fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"../results/e3/batch{batch_number}_scores.png")
    plt.show()

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

batch_scores = {"f1": [], "rec": [], "prec": [], "acc": [], "bac": [], "gmean": [], "spec": []}
epoch_scores = {"f1": [], "rec": [], "prec": [], "acc": [], "bac": [], "gmean": [], "spec": []}


def get_scores(y_true, y_pred):
    batch_scores["f1"].append(f1_score(y_true, y_pred))
    batch_scores["rec"].append(recall_score(y_true, y_pred))
    batch_scores["prec"].append(precision_score(y_true, y_pred))
    batch_scores["acc"].append(accuracy_score(y_true, y_pred))
    batch_scores["bac"].append(balanced_accuracy_score(y_true, y_pred))
    batch_scores["gmean"].append(geometric_mean_score(y_true, y_pred))
    batch_scores["spec"].append(specificity_score(y_true, y_pred))


def get_epoch_scores(scores: dict):
    for key, score in scores.items():
        epoch_scores[key].append(np.mean(score))

    # scores = {"f1": [], "rec": [], "prec": [], "acc": [], "bac": [], "gmean": [], "spec": []}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

labels_file = "../binaryModel/archive/labels.csv"
labels_df = pd.read_csv(labels_file)
labels = labels_df['sick'].values
dataset = ImageDataset(img_dir='../img/archive/all_images', labels_file=labels_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=50)

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{epochs}")

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            get_scores(y_true, y_pred)

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % 5 == 0:
        plot_batch_scores(batch_scores, data_loader, epoch)

    get_epoch_scores(batch_scores)
    batch_scores = {"f1": [], "rec": [], "prec": [], "acc": [], "bac": [], "gmean": [], "spec": []}
    print(f"Loss: {running_loss / len(data_loader):.4f}")

print(epoch_scores)

# for key, score in epoch_scores.items():
#     plt.plot(score, label=key)
#     plt.savefig(f"{key}_deep.png", dpi=200)
#     plt.show()

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))
axes = axes.ravel()

metric_keys = ["f1", "rec", "prec", "acc", "bac", "gmean", "spec"]

for i, metric in enumerate(metric_keys):
    axes[i].plot(epoch_scores[metric], marker='None', label=metric)
    axes[i].set_title(score_names[i])
    axes[i].set_xlabel('Epoka')
    axes[i].set_ylabel('Średni wynik')
    axes[i].grid(True)

for j in range(len(metric_keys), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("../results/e3/scores.png")
plt.show()

print("Training and evaluation completed.")