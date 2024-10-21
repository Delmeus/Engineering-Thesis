import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        # Load the labels from the CSV file
        self.labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the image file name and label from the labels DataFrame
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])  # Get image filename
        label = self.labels.iloc[idx, 1]  # Get corresponding label

        # Load the image using PIL
        image = Image.open(img_name).convert("RGB")  # Convert to RGB (for 3-channel images)

        # Apply transforms (e.g., resizing, normalization) if provided
        if self.transform:
            image = self.transform(image)

        return image, label
