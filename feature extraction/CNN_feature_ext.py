import os
import tensorflow.keras.applications.vgg16 as vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

# Define the path to the folder containing the image classes
resized_folder_path = "venv/resized"

# 1. Load the model with a smaller output layer
model = vgg16.VGG16(weights="imagenet", include_top=False)
# Get features from an earlier layer with fewer output units
model = Model(inputs=model.input, outputs=model.get_layer(index=-4).output)  # Example using -4


# 3. Refactor code for efficiency
CNN_features = []
for folder_name in os.listdir(resized_folder_path):
    folder_path = os.path.join(resized_folder_path, folder_name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x).flatten()  # Get features directly
        CNN_features.append(features)

# 4. Save features to a compressed CSV file
df = pd.DataFrame(CNN_features)
df.to_csv("CNN_features.csv.gz", compression="gzip")
