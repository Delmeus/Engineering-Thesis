import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG156
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

images_path = '../../../img/archive/all_images'
labels_csv_path = '../../archive/labels.csv'

labels_df = pd.read_csv(labels_csv_path)

image_filenames = labels_df['image'].values
labels = labels_df['sick'].values

# Initialize VGG16 model for feature extraction (without the top classification layer)
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

image_features = []
image_labels = []

for img_name, label in zip(image_filenames, labels):
    img_path = os.path.join(images_path, img_name)
    img_path = os.path.normpath(img_path) + '.jpg'

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)

    # Preprocess the image for VGG16
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features
    features = model.predict(img_array)
    features = features.flatten()

    image_features.append(features)
    image_labels.append(label)

image_features = np.array(image_features)
image_labels = np.array(image_labels)

dataset = np.column_stack((image_features, image_labels))

np.save('../../../npy_datasets/image_dataset.npy', dataset)

print("Dataset saved as image_dataset.npy")
