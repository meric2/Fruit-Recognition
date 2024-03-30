import os

base_dir = ""

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "validation")

class_names = os.listdir(train_dir)

num_of_classes = len([subdir for subdir in os.listdir(train_dir)])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import resize_with_crop_or_pad
import tensorflow as tf


def crop_to_target_size(image):
    # TensorFlow'un resize_with_crop_or_pad fonksiyonu, hedef boyutu geçtiğinde kırpma, eksik olduğunda doldurma yapar.
    # Burada, resmi her zaman hedef boyuta kırpıyoruz.
    return resize_with_crop_or_pad(image, target_height=128, target_width=128)


# ImageDataGenerator nesnelerini güncelleme
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    preprocessing_function=crop_to_target_size,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    preprocessing_function=crop_to_target_size,
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    preprocessing_function=crop_to_target_size,
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=50,
    class_mode="categorical",
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=50, class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=50, class_mode="categorical"
)

# Construct the model
import tensorflow as tf
from tensorflow.keras import models, layers

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(128, 128, 3)
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_of_classes, activation="softmax"),
    ]
)


from tensorflow.keras.optimizers import RMSprop, Adam, SGD

optimizer = SGD(learning_rate=0.001, momentum=0.9)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=150,
    epochs=128,
    validation_data=test_generator,
    validation_steps=50,
    verbose=1,
)


model.save("CNN_model.h5")

# save CNN architecture as text file
with open("CNN_model_architecture.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# save layers as text file
with open("CNN_model_layers.txt", "w") as f:
    for layer in model.layers:
        f.write(str(layer) + "\n")

import matplotlib.pyplot as plt

# Assuming 'history' is the variable containing the training history
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

# Plotting training & validation accuracy
# Adjusted plotting to handle unequal length arrays
min_epochs = min(len(epochs), len(val_accuracy))  # Find the minimum length
plt.plot(epochs[:min_epochs], accuracy[:min_epochs], "bo", label="Training Accuracy")
plt.plot(
    epochs[:min_epochs], val_accuracy[:min_epochs], "b", label="Validation Accuracy"
)
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig("training_validation_accuracy.png")
