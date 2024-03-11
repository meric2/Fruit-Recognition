import os
import cv2
import numpy as np
import mahotas as mh

def calculate_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def calculate_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = mh.features.haralick(gray_image).mean(axis=0)

    return features


def calculate_dominant_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color = np.unravel_index(hist.argmax(), hist.shape)
    return color


def feature_extraction(folder):
    texture_features = []
    histograms = []
    dominant_colors = []
    labels = []

    # Iterate over the subfolders in the folder
    for folder_name in os.listdir(folder):
        folder_path = os.path.join(folder, folder_name)
        
        # Iterate over the images in each subfolder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            
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
    np.savetxt(folder+"_texture_features.csv", texture_features, delimiter=",")
    np.savetxt(folder+"_histograms.csv", histograms, delimiter=",")
    np.savetxt(folder+"_dominant_colors.csv", dominant_colors, delimiter=",")
    np.savetxt(folder+"_labels.csv", labels, delimiter=",", fmt="%s")

feature_extraction("train")
print("Features extracted for training images")
feature_extraction("test")
print("Features extracted for test images")
feature_extraction("validation")



