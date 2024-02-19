import os
import cv2
import numpy as np
import mahotas as mh

def calculate_color_histogram(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def calculate_texture_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Haralick texture features
    features = mh.features.haralick(gray_image).mean(axis=0)

    return features


def calculate_dominant_color(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Find the most dominant color
    color = np.unravel_index(hist.argmax(), hist.shape)

    return color



resized_folder_path = "venv/resized"

texture_features = []
histograms = []
dominant_colors = []
labels = []

# Iterate over the subfolders in the resized_folder_path
for folder_name in os.listdir(resized_folder_path):
    folder_path = os.path.join(resized_folder_path, folder_name)
    
    # Iterate over the images in each subfolder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Calculate the texture features
        features = calculate_texture_features(image)
        histogram = calculate_color_histogram(image)
        color = calculate_dominant_color(image)

        texture_features.append(features)
        histograms.append(histogram)
        dominant_colors.append(color)

        labels.append(folder_name)

# Convert the lists to NumPy arrays
histograms = np.array(histograms)
texture_features = np.array(texture_features)
dominant_colors = np.array(dominant_colors)


# Save the features to CSV files
np.savetxt("train_texture_features.csv", texture_features, delimiter=",")
np.savetxt("train_histograms.csv", histograms, delimiter=",")
np.savetxt("train_dominant_colors.csv", dominant_colors, delimiter=",")

# Save the labels to a csv file
np.savetxt("train_labels.csv", labels, delimiter=",", fmt="%s")
