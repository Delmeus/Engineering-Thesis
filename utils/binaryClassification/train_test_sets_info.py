import pandas as pd

def print_healthy_percentage(df, type : str = ""):
    total_images = len(df)
    healthy_images = df['healthy'].sum()
    percentage_healthy = (healthy_images / total_images) * 100
    print(f"Percentage of healthy images in {type} files: {percentage_healthy:.2f}% and there are {total_images} images")

train_values = '../../img/archive/train_values.csv'
test_values = '../../img/archive/test_values.csv'

df_train = pd.read_csv(train_values)
df_test = pd.read_csv(test_values)

print_healthy_percentage(df_train, "train")
print_healthy_percentage(df_test, "test")
