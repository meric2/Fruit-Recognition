import os
import random
import shutil

# Set the base directory
base_dir = ""

# Create the train, test, and valid directories
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "valid")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Set the data directory
data_dir = os.path.join(base_dir, "data")

# Get the list of subdirectories in the data directory
subdirectories = [
    subdir
    for subdir in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, subdir))
]

# Iterate over each subdirectory
for subdir in subdirectories:
    # Get the list of images in the subdirectory
    images = [
        image
        for image in os.listdir(os.path.join(data_dir, subdir))
        if image.endswith(".jpg")
    ]

    # Shuffle the images randomly
    random.shuffle(images)

    # Calculate the number of images for each split
    num_images = len(images)
    num_train = int(0.8 * num_images)
    num_test = (num_images - num_train) // 2
    num_valid = num_images - num_train - num_test

    # Create the subdirectories in the train, test, and valid directories
    os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, subdir), exist_ok=True)

    # Split the images into train, test, and valid directories
    for i, image in enumerate(images):
        if i < num_train:
            shutil.copy2(
                os.path.join(data_dir, subdir, image),
                os.path.join(train_dir, subdir, image),
            )
        elif i < num_train + num_test:
            shutil.copy2(
                os.path.join(data_dir, subdir, image),
                os.path.join(test_dir, subdir, image),
            )
        else:
            shutil.copy2(
                os.path.join(data_dir, subdir, image),
                os.path.join(valid_dir, subdir, image),
            )
