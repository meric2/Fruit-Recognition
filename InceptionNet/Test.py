import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


test_dir = "data_128x128/test"
batch_size = 32
IMG_SIZE = (150, 150)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

class_names = test_dataset.class_names

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Evaluate the model on the test data using `evaluate`
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=50000,  # Loading all data at once
    class_mode="categorical",
    shuffle=False,
)

# Load the model
model = tf.keras.models.load_model("InceptionNet.h5")

# Compile the model with the desired metrics
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)

print("\nTest accuracy: {:.2f}".format(test_acc))
print("\nTest loss: {:.2f}".format(test_loss))

# Calculate predictions
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

from sklearn.metrics import classification_report

print("Classification Report")
print(classification_report(test_generator.classes, y_pred, target_names=class_names))

# Calculate confusion matrix
confusion_mtx = confusion_matrix(test_generator.classes, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

# Adding labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
# Show the plot
plt.show()
