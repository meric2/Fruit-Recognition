import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.python.platform.build_info as build
import tensorflow_hub as hub

print("TensorFlow version: ", tf.__version__)
print("Eager execution: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print(
    "GPU is",
    (
        "available"
        if tf.config.experimental.list_physical_devices("GPU")
        else "NOT AVAILABLE"
    ),
)
print("CUDA version: ", build.build_info["cuda_version"])
print("CUDNN version: ", build.build_info["cudnn_version"])
