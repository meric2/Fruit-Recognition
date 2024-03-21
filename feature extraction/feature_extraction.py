import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


def get_lbp_image(gray_image):
    lbp_image = np.zeros_like(gray_image)
    neigh = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    for i in range(0, gray_image.shape[0] - 2):
        for j in range(0, gray_image.shape[1] - 2):
            window = gray_image[i : i + 3, j : j + 3]
            lbp_image[i + 1, j + 1] = np.sum(window * neigh >= 0)

    return lbp_image


def calculate_lbp_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = get_lbp_image(gray_image)
    # Compute the histogram with fewer bins to reduce dimensionality
    hist, _ = np.histogram(lbp_image.ravel(), bins=64, range=(0, 256))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7  # Normalize the histogram

    return hist


def calculate_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate LBP features using get_lbp_image
    lbp_image = get_lbp_image(gray_image)

    # Extract statistical features from LBP (e.g., mean, variance)
    mean_lbp = np.mean(lbp_image)
    var_lbp = np.var(lbp_image)

    feature_vector = [mean_lbp, var_lbp]
    return feature_vector


def calculate_dominant_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    color = np.unravel_index(hist.argmax(), hist.shape)
    return color


def feature_extraction(folder):
    texture_features = []
    histograms = []
    dominant_colors = []
    labels = []

    # Iterate over the subfolders in the folder
    for folder_name in tqdm(os.listdir(folder)):
        folder_path = os.path.join(folder, folder_name)

        # Iterate over the images in each subfolder
        for image_name in tqdm(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            features = calculate_texture_features(image)
            histogram = calculate_lbp_histogram(image)
            color = calculate_dominant_color(image)

            texture_features.append(features)
            histograms.append(histogram)
            dominant_colors.append(color)
            labels.append(folder_name)

    texture_df = pd.DataFrame(texture_features, columns=["mean_lbp", "var_lbp"])
    histogram_df = pd.DataFrame(histograms)
    color_df = pd.DataFrame(dominant_colors, columns=["H", "S", "V"])
    label_df = pd.DataFrame(labels, columns=["label"])

    texture_df.to_csv("texture_features.csv", index=False)
    histogram_df.to_csv("histograms.csv", index=False)
    color_df.to_csv("dominant_colors.csv", index=False)
    label_df.to_csv("labels.csv", index=False)


feature_extraction("data")
