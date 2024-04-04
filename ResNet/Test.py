import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# Load the model
model = tf.keras.models.load_model("ResNet_model.h5")

# Define the directory containing the test images
test_dir = "test"

# Define the batch size and image size, consistent with the training phase
batch_size = 32
IMG_SIZE = (160, 160)

# Prepare the test dataset using the same method as the training and validation datasets
# Prepare the test dataset using the same method as the training and validation datasets
test_dataset_raw = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=False,  # It's better not to shuffle for testing to keep labels ordered
    batch_size=batch_size,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

# Capture class names from the raw dataset
class_names = test_dataset_raw.class_names

# Use the AUTOTUNE feature to optimize loading
AUTOTUNE = tf.data.AUTOTUNE
test_dataset = test_dataset_raw.prefetch(buffer_size=AUTOTUNE)


# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_dataset)
print("\nTest accuracy: {:.2f}%".format(accuracy * 100))
print("Test loss: {:.2f}".format(loss))

# Generate predictions
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Prepare true labels
true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
y_true = np.argmax(true_labels, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,  # Use captured class names here
    yticklabels=class_names,  # And here
)

# Adding labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
