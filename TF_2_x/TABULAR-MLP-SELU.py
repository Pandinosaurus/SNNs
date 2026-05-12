from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation="selu", kernel_initializer="lecun_normal"),
        layers.AlphaDropout(0.05),
        layers.Dense(32, activation="selu", kernel_initializer="lecun_normal"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
print(model.evaluate(X_test, y_test, verbose=0))