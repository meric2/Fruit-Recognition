import os
import shutil
import random

# Define the path to your dataset
dataset_path = "venv/fruit_dataset"

# Define the ratio for train, test, validation split
train_ratio = 0.7
test_ratio = 0.2
validation_ratio = 0.1

# Define the paths for train, test, validation folders
train_path = "venv/train"
test_path = "venv/test"
validation_path = "venv/validation"

# Create directories for train, test, validation splits
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(validation_path, exist_ok=True)

# Iterate over each class folder
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        # Create corresponding class folders in train, test, validation splits
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(validation_path, class_name), exist_ok=True)
        
        # Get list of images in the class folder
        images = os.listdir(class_path)
        # Shuffle the images randomly
        random.shuffle(images)
        
        # Split the images into train, test, validation sets
        train_split = int(len(images) * train_ratio)
        test_split = int(len(images) * test_ratio)
        validation_split = int(len(images) * validation_ratio)
        
        # Assign images to train, test, validation folders
        for img in images[:train_split]:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_path, class_name))
        for img in images[train_split:train_split+test_split]:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_path, class_name))
        for img in images[train_split+test_split:train_split+test_split+validation_split]:
            shutil.copy(os.path.join(class_path, img), os.path.join(validation_path, class_name))
