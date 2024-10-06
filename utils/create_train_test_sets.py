import os
import random
import shutil


def move_files(source_dir, destination_dir, percentage=0.8):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)  # Create the destination directory if it doesn't exist

    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    random.shuffle(files)

    num_files_to_move = int(len(files) * percentage)

    files_moved = 0
    for file in files[:num_files_to_move]:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        shutil.move(source_file, destination_file)
        files_moved += 1

    print(f"Moved {files_moved} files from '{source_dir}' to '{destination_dir}'.")

source_directory = "/home/antek/studia/inzynierka/Engineering-Thesis/img/archive/test"
destination_directory = "/home/antek/studia/inzynierka/Engineering-Thesis/img/archive/train"

move_files(source_directory, destination_directory, percentage=0.8)
