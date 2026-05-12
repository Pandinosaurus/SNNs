import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


batch_size = 128
num_classes = 10
epochs = 5
learning_rate = 0.001

# List devices so you can check whether your GPU is available.
print(tf.config.list_physical_devices())

# The data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255.0
x_test = x_test[..., None].astype("float32") / 255.0

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

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(
            32,
            (3, 3),
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
        layers.AlphaDropout(0.05),
        layers.Flatten(),
        layers.Dense(
            512,
            activation="selu",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
        layers.AlphaDropout(0.05),
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_initializer=keras.initializers.LecunNormal(),
            bias_initializer="zeros",
        ),
    ]
)

# Compile the model with the appropriate loss function, optimizer, and metrics.
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)

# Train the model and validate on the validation set.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)

# Evaluate the model on the test set.
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
