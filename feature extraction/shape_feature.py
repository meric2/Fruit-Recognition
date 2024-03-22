import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


import cv2
import numpy as np


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


def feature_extraction(folder):
    shape_features = []

    # Iterate over the subfolders in the folder
    for folder_name in tqdm(os.listdir(folder)):
        folder_path = os.path.join(folder, folder_name)

        # Iterate over the images in each subfolder
        for image_name in tqdm(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            shape_feature = extract_shape_features(image)
            shape_features.append(shape_feature)

    df = pd.DataFrame(
        shape_features,
        columns=["Area", "Perimeter", "Aspect Ratio", "Extent", "Solidity"],
    )
    df.to_csv("shape_features.csv", index=False)


feature_extraction("data")
