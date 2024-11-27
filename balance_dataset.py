import numpy as np
from imblearn.over_sampling import SMOTE

def load_npy_dataset(file_path):
    """
    Load dataset from an .npy file.
    Assumes the data is saved in the format (X, y), where:
      - X: Features (2D array or needs reshaping)
      - y: Labels (1D array)
    """
    data = np.load(file_path, allow_pickle=True)
    X, y = data[0], data[1]

    # Check if X is 1D and reshape if necessary
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)  # Reshape to (n_samples, 1)

    return X, y

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to the dataset.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def save_npy_dataset(X, y, file_path):
    """
    Save the dataset to an .npy file in the format (X, y).
    """
    np.save(file_path, (X, y))

# Load the dataset
file_path = "npy_datasets/hog_dataset.npy"
X, y = load_npy_dataset(file_path)

# Check unique values in y
print("Unique labels in y before processing:", np.unique(y))

# Convert continuous labels to discrete classes
y = (y > 0.5).astype(int)  # Example threshold; adjust as needed
print("Unique labels in y after processing:", np.unique(y))

# Check shapes
print("Shape of X before SMOTE:", X.shape)
print("Shape of y before SMOTE:", y.shape)

# Apply SMOTE
X_resampled, y_resampled = apply_smote(X, y)

# Check shapes after SMOTE
print("Shape of X after SMOTE:", X_resampled.shape)
print("Shape of y after SMOTE:", y_resampled.shape)

# Save the new dataset
output_file_path = "npy_datasets/hog_balanced_dataset.npy"
save_npy_dataset(X_resampled, y_resampled, output_file_path)

print(f"SMOTE applied and resampled dataset saved to {output_file_path}")
