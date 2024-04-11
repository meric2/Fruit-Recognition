# https://www.tensorflow.org/tutorials/images/transfer_learning?hl=tr

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

base_dir = "data_128x128"

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")


batch_size = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

validation_dateset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=IMG_SIZE,
    label_mode="categorical",
)


class_names = train_dataset.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dateset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]
)

preprocess_input = tf.keras.applications.resnet.preprocess_input
rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

IMG_SIZE = IMG_SIZE + (3,)  # add the channel dimension

base_model = tf.keras.applications.ResNet50(
    input_shape=IMG_SIZE, include_top=False, weights="imagenet"
)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False  # freeze the base model for transfer learning


base_model.summary()  # print the summary of the base model

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

num_of_classes = len(class_names)

prediction_layer = tf.keras.layers.Dense(num_of_classes, activation="softmax")
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.15)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

print(
    "Number of training batches: %d" % tf.data.experimental.cardinality(train_dataset)
)
print(
    "Number of validation batches: %d"
    % tf.data.experimental.cardinality(validation_dataset)
)


base_learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

len(model.trainable_variables)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


early_stopping = [
    EarlyStopping(monitor="val_loss", patience=20),
    ModelCheckpoint(filepath="ResNet.h5", monitor="val_loss", save_best_only=True),
]

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping],
    verbose=1,
    steps_per_epoch=len(train_dataset) // batch_size,
    validation_steps=20,
)
