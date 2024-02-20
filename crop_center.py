"""
    Crop the central part of the image and resize it to the specified size.

    Parameters:
    - image: The input image as a NumPy array.
    - output_size: A tuple specifying the desired output size (width, height).

    Returns:
    - The cropped and resized image as a NumPy array.
    """

import cv2
import os
import numpy as np

def crop_(image, output_size=(32, 32)):
    height, width = image.shape[:2]
    
    # Determine the central crop size
    crop_size = min(height, width)
    
    # Calculate crop margins
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    
    # Perform the central crop
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # Resize the cropped image to the output size
    resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_AREA)
    
    return resized_image


def walk_images_in_folder(input_folder, output_folder, new_width, new_height):
    for root, _, _ in os.walk(input_folder):# iterate through train, test, and validation folders
        for subdir in os.listdir(root):
            # Create corresponding output directory
            output_subdir = os.path.join(output_folder, os.path.relpath(root, input_folder), subdir)
            os.makedirs(output_subdir, exist_ok=True)
            # Resize images in each class folder
            for filename in os.listdir(os.path.join(root, subdir)):
                input_path = os.path.join(root, subdir, filename)
                output_path = os.path.join(output_subdir, filename)
                # Read the image
                image = cv2.imread(input_path)
                # Resize the image using seam carving
                resized_image = crop_(image, (new_width, new_height))
                # Save the resized image
                cv2.imwrite(output_path, resized_image)


# Change the paths to the corresponding folders
# Code handles a specific folder structure e.g. input_folder/class_name/image.jpg 
input_folder = 'data/test'
output_folder = 'centralized_data/test'
new_width = 32
new_height = 32

walk_images_in_folder(input_folder, output_folder, new_width, new_height)
