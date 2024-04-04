import os

base_dir = ""

test_dir = os.path.join(base_dir, "test")

class_names = os.listdir(test_dir)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import resize_with_crop_or_pad

INPUT_SHAPE = (150, 150)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=INPUT_SHAPE,
    batch_size=50,
    class_mode="categorical",
    shuffle=False,
)


import tensorflow as tf

model = tf.keras.models.load_model("CNN_model.h5")

# Evaluate the model on the test data and print the results and plot the confusion matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

test_loss, test_acc = model.evaluate(test_generator)
print("\nTest accuracy:", test_acc)

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("Classification Report")
target_names = class_names
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot the confusion matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

plt.show()
