import os
import sys

import pandas as pd

def check_sickness_status(directory, csv_file, output_file):
    df = pd.read_csv(csv_file)

    df['image'] = df['image'].apply(lambda x: os.path.splitext(x)[0])

    with open(output_file, 'w') as f:
        f.write('filename,healthy\n')

        for filename in os.listdir(directory):
            file_without_ext = os.path.splitext(filename)[0]

            if file_without_ext in df['image'].values:
                healthy_status = df.loc[df['image'] == file_without_ext, 'healthy'].values[0]
                f.write(f'{file_without_ext},{healthy_status}\n')

    print(f"Output written to {output_file}")

csv_file = 'img/archive/binaryProblem.csv'

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_directory> <destination_directory>")
        return

    dir = sys.argv[1]
    output_file = dir + '/' + sys.argv[2]

    if not os.path.exists(dir):
        print("Podana sciezka nie istnieje lub nie jest katalogiem")
        return

    check_sickness_status(dir, csv_file, output_file)

main()
