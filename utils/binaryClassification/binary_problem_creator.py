import pandas as pd

input_file = 'img/archive/GroundTruth.csv'
output_file = 'img/archive/binaryProblem.csv'

df = pd.read_csv(input_file)

df['healthy'] = 1

not_healthy_conditions = ['MEL', 'BCC', 'AKIEC']

# BKL is healthy? (not cancer)
# DF is not cancer
# NV is healthy
# VASC - debatable, rather healthy

for condition in not_healthy_conditions:
    df.loc[df[condition] == 1.0, 'healthy'] = 0

df = df[['image', 'healthy']]

df.to_csv(output_file, index=False)

print(f"New CSV file saved as {output_file}")
