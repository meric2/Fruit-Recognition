import os
from PIL import Image

# Define the path to the folder containing the image classes
folders_path = "venv/train"

# Define the path to the folder where resized images will be saved
resized_folder_path = "venv/resized"

# Create the resized folder if it doesn't exist
os.makedirs(resized_folder_path, exist_ok=True)

# Iterate over the subfolders in the folders_path
for folder_name in os.listdir(folders_path):
    folder_path = os.path.join(folders_path, folder_name)
    
    # Iterate over the images in each subfolder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Open the image
        image = Image.open(image_path)
        
        # Resize the image to 128x128
        resized_image = image.resize((128, 128))
        # Convert the image to RGB mode
        resized_image = resized_image.convert("RGB")

        # Save the resized image to the resized folder
        resized_image_path = os.path.join(resized_folder_path, folder_name, image_name)
        os.makedirs(os.path.dirname(resized_image_path), exist_ok=True)
        resized_image.save(resized_image_path)