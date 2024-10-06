import os
import sys
def get_filenames_without_extension(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []

    filenames = [os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    return filenames

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <source_directory> <destination_directory>")
        return

    src = sys.argv[1]

    if not os.path.exists(src):
        print("Podana sciezka nie istnieje lub nie jest katalogiem")
        return

    filenames = get_filenames_without_extension(src)

    for filename in filenames:
        print(filename)

main()