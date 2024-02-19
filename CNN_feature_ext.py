import os
import tensorflow.keras.applications.vgg16 as vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# Define the path to the folder containing the image classes
resized_folder_path = "venv/resized"

# Load the VGG16 model pre-trained on ImageNet data
model = vgg16.VGG16(weights="imagenet", include_top=False)

# Get the features from the second last layer. 
model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
 

# Create empty lists to store the features and labels
CNN_features = []

# Iterate over the subfolders in the resized_folder_path
for folder_name in os.listdir(resized_folder_path):
    folder_path = os.path.join(resized_folder_path, folder_name)
    
    # Iterate over the images in each subfolder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Load the image in the correct format for VGG16
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Get the features from the VGG16 model
        CNN_features.append(model.predict(x).flatten())

# Convert the list to a NumPy array
CNN_features = np.array(CNN_features)

#save the features to a csv file
np.savetxt("CNN_features.csv", CNN_features, delimiter=",")
