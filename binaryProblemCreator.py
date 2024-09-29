import pandas as pd

# Load the CSV file
input_file = 'GroundTruth.csv'
output_file = 'binaryProblem.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Create a new column 'healthy' with default value 0 (healthy)
df['healthy'] = 0

# Define the conditions to classify as 'not healthy'
not_healthy_conditions = ['MEL', 'BCC', 'AKIEC', 'BKL', 'VASC']

# Iterate through the rows and classify as 'not healthy' (1) if any of the conditions is present
for condition in not_healthy_conditions:
    df.loc[df[condition] == 1.0, 'healthy'] = 1

# Create the 'not healthy' column which is the inverse of the 'healthy' column
df['sick'] = 1 - df['healthy']

# Keep only the 'image', 'healthy', and 'not healthy' columns
df = df[['image', 'healthy', 'sick']]

# Save the new CSV file
df.to_csv(output_file, index=False)

print(f"New CSV file saved as {output_file}")
