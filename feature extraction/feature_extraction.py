import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


def extract_shape_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to hold the feature values for each contour
    areas = []
    perimeters = []
    aspect_ratios = []
    extents = []
    solidities = []

    for contour in contours:
        # Calculate area of the contour and add to list
        area = cv2.contourArea(contour)
        areas.append(area)

        # Calculate the perimeter of the contour and add to list
        perimeter = cv2.arcLength(contour, True)
        perimeters.append(perimeter)

        # Calculate aspect ratio and add to list
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        aspect_ratios.append(aspect_ratio)

        # Calculate extent and add to list
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        extents.append(extent)

        # Calculate solidity and add to list
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        solidities.append(solidity)

    # If there are no contours, return zeros for all features
    if len(contours) == 0:
        return [0, 0, 0, 0, 0]

    # Calculate the mean of each feature across all contours
    mean_area = np.mean(areas)
    mean_perimeter = np.mean(perimeters)
    mean_aspect_ratio = np.mean(aspect_ratios)
    mean_extent = np.mean(extents)
    mean_solidity = np.mean(solidities)

    # Return a single set of features representing the mean values
    return [mean_area, mean_perimeter, mean_aspect_ratio, mean_extent, mean_solidity]


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
    shape_features = []
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
            shape_feature = extract_shape_features(image)

            shape_features.append(shape_feature)
            texture_features.append(features)
            histograms.append(histogram)
            dominant_colors.append(color)
            labels.append(folder_name)

    texture_df = pd.DataFrame(texture_features, columns=["mean_lbp", "var_lbp"])
    histogram_df = pd.DataFrame(histograms)
    color_df = pd.DataFrame(dominant_colors, columns=["H", "S", "V"])
    label_df = pd.DataFrame(labels, columns=["label"])
    shape_df = pd.DataFrame(
        shape_features,
        columns=["Area", "Perimeter", "Aspect Ratio", "Extent", "Solidity"],
    )

    shape_df.to_csv("shape_features.csv", index=False)
    texture_df.to_csv("texture_features.csv", index=False)
    histogram_df.to_csv("histograms.csv", index=False)
    color_df.to_csv("dominant_colors.csv", index=False)
    label_df.to_csv("labels.csv", index=False)


def folder_feature_extraction(folder):
    texture_features = []
    histograms = []
    dominant_colors = []
    shape_features = []
    labels = []
    image_paths = []
    for image_name in tqdm(os.listdir(folder)):
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)

        features = calculate_texture_features(image)
        histogram = calculate_lbp_histogram(image)
        color = calculate_dominant_color(image)
        shape_feature = extract_shape_features(image)

        shape_features.append(shape_feature)
        texture_features.append(features)
        histograms.append(histogram)
        dominant_colors.append(color)
        image_paths.append(image_path)
        labels.append(folder)

    texture = pd.DataFrame(texture_features, columns=["mean_lbp", "var_lbp"])
    histogram = pd.DataFrame(histograms)
    dominant_color = pd.DataFrame(dominant_colors, columns=["H", "S", "V"])
    label = pd.DataFrame(labels, columns=["label"])
    shape = pd.DataFrame(
        shape_features,
        columns=["Area", "Perimeter", "Aspect Ratio", "Extent", "Solidity"],
    )
    path = pd.DataFrame(image_paths, columns=["path"])

    batch_df = pd.concat(
        [texture, dominant_color, histogram, shape, label, path], axis=1
    )
    batch_df.to_csv("batch_features.csv", index=False)


# feature_extraction("data")
folder_feature_extraction("cocoa bean")
