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
from scipy.stats import gmean
from binaryModel.utils.ImageDataset import ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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

num_classes = 2
model = CNNModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


batch_f1_scores = []
batch_f1_scores_all = []

epochs = 10
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
            batch_f1_scores.append(f1_score(y_true, y_pred, average='macro'))

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    plt.plot(batch_f1_scores)
    batch_f1_scores_all.append(np.mean(batch_f1_scores))
    batch_f1_scores = []
    plt.show()
    print(f"Loss: {running_loss / len(data_loader):.4f}")

plt.plot(batch_f1_scores_all)
plt.show()

print("Training and evaluation completed.")
