"""
Resizes images to wanted size.
Uses seam carving while doing so to not lose too many features.
"""



import os
import cv2
import numpy as np

def seam_carving(image, new_width, new_height):
    current_width, current_height = image.shape[1], image.shape[0]
    delta_width = current_width - new_width
    delta_height = current_height - new_height

    # Calculate energy map of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = cv2.magnitude(sobel_x, sobel_y)

    # Remove vertical seams
    if delta_width < 0:
        for i in range(abs(delta_width)):
            cumulative_energy_map = cv2.copyMakeBorder(energy_map, 1, 1, 0, 0, cv2.BORDER_REPLICATE)
            for j in range(1, current_height):
                for k in range(current_width):
                    cumulative_energy_map[j, k] += min(cumulative_energy_map[j-1, max(k-1, 0):min(k+2, current_width)].min(), cumulative_energy_map[j-1, k])
            mask = np.ones((current_height, current_width), dtype=bool)
            j = np.argmin(cumulative_energy_map[-1])
            for i in reversed(range(current_height)):
                mask[i, j] = False
                j -= 1 if j == 0 else np.argmin(cumulative_energy_map[i-1, max(j-1, 0):min(j+2, current_width)]) - 1
            current_width -= 1
            image = image[mask].reshape((current_height, current_width, 3))

    # Remove horizontal seams
    if delta_height < 0:
        image = np.rot90(image, 1, (0, 1))
        energy_map = cv2.transpose(energy_map)
        energy_map = cv2.flip(energy_map, 1)
        for i in range(abs(delta_height)):
            cumulative_energy_map = cv2.copyMakeBorder(energy_map, 0, 0, 1, 1, cv2.BORDER_REPLICATE)
            for j in range(1, current_width):
                for k in range(current_height):
                    cumulative_energy_map[k, j] += min(cumulative_energy_map[max(k-1, 0):min(k+2, current_height), j-1].min(), cumulative_energy_map[k, j-1])
            mask = np.ones((current_height, current_width), dtype=bool)
            j = np.argmin(cumulative_energy_map[:, -1])
            for i in reversed(range(current_width)):
                mask[j, i] = False
                j -= 1 if j == 0 else np.argmin(cumulative_energy_map[max(j-1, 0):min(j+2, current_height), i-1]) - 1
            current_height -= 1
            image = image[mask].reshape((current_width, current_height, 3))
        image = np.rot90(image, 3, (0, 1))

    return cv2.resize(image, (new_width, new_height))


def resize_images_in_folder(input_folder, output_folder, new_width, new_height):
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
                resized_image = seam_carving(image, new_width, new_height)
                # Save the resized image
                cv2.imwrite(output_path, resized_image)


# Change the paths to the corresponding folders
# Code handles a specific folder structure e.g. input_folder/class_name/image.jpg 
input_folder = 'redimmed_64x64/train'
output_folder = 'redimmed_64x64/train'
new_width = 64
new_height = 64

resize_images_in_folder(input_folder, output_folder, new_width, new_height)
