import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


batch_size = 32
num_classes = 10
epochs = 2
data_augmentation = True
num_predictions = 20

# List devices so you can check whether your GPU is available.
print(tf.config.list_physical_devices())

# The data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create validation set.
x_val = x_train[:10000]
x_train = x_train[10000:]
y_val = y_train[:10000]
y_train = y_train[10000:]

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_val.shape[0], "val samples")
print(x_test.shape[0], "test samples")

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model_layers = [keras.Input(shape=x_train.shape[1:])]

if data_augmentation:
    print("Using data augmentation.")
    model_layers.extend(
        [
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip("horizontal"),
        ]
    )
else:
    print("Not using data augmentation.")

model_layers.extend(
    [
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="selu",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
        layers.Conv2D(
            32,
            (3, 3),
            activation="selu",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.AlphaDropout(0.1),
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="selu",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
        layers.Conv2D(
            64,
            (3, 3),
            activation="selu",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.AlphaDropout(0.1),
        layers.Flatten(),
        layers.Dense(
            512,
            activation="selu",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
        layers.AlphaDropout(0.2),
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
    ]
)

model = keras.Sequential(model_layers)

# Compile the model with the appropriate loss function, optimizer, and metrics.
optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=["accuracy"],
)

# Train the model and validate on the validation set.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    shuffle=True,
)

# Evaluate the model on the test set.
evaluation = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print(f"Model Accuracy = {evaluation[1]:.5f}")

# Generate and print predictions for the test set.
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
predictions = model.predict(x_test[: num_predictions + 1], batch_size=batch_size)
for predict_index, predicted_y in enumerate(predictions):
    actual_label = class_names[np.argmax(y_test[predict_index])]
    predicted_label = class_names[np.argmax(predicted_y)]
    print(f"Actual Label = {actual_label} vs. Predicted Label = {predicted_label}")
