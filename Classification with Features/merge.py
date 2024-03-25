import os
import shutil

# Set the paths for the test, train, and validation directories
test_dir = "test"
train_dir = "train"
validation_dir = "validation"

# Set the path for the data directory
data_dir = "data"

# Get the list of folder names in the test directory
test_folders = os.listdir(test_dir)

# Iterate over each folder name
for folder_name in test_folders:
    # Check if the folder exists in the train and validation directories
    if folder_name in os.listdir(train_dir) and folder_name in os.listdir(
        validation_dir
    ):
        # Create the destination folder in the data directory
        dest_folder = os.path.join(data_dir, folder_name)
        os.makedirs(dest_folder, exist_ok=True)

        # paste the images from the test, train, and validation directories to the data directory
        for directory in [test_dir, train_dir, validation_dir]:
            src_folder = os.path.join(directory, folder_name)
            for file_name in os.listdir(src_folder):
                src_file = os.path.join(src_folder, file_name)
                dest_file = os.path.join(dest_folder, file_name)
                shutil.copy(src_file, dest_file)
