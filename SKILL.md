---
name: self-normalizing-networks
description: Use this skill when working in the Self-Normalizing Networks tutorial repository, especially for TensorFlow, PyTorch, Conda environment, or SELU/AlphaDropout implementation tasks.
---

# Self-Normalizing Networks

This repository contains tutorial implementations for Self-Normalizing Networks
(SNNs) based on Klambauer et al. Preserve the teaching purpose of the examples:
show how SELU networks are constructed and compared across TensorFlow and
PyTorch implementations.

## Minimal Tabular SNN Example

Below is a minimal prototype (see TF_2_x/TABULAR-MLP-SELU.py) for showing SNNs 
on tabular data: load a small scikit-learn dataset, standardize the features, 
and train a compact Keras MLP with SELU, LeCun initialization, and AlphaDropout.

```python
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
```

## Core SNN Rules

- Pair SELU activations with LeCun normal initialization.
- Use AlphaDropout when dropout is used in SELU networks.
- In PyTorch, use `nn.init.kaiming_normal_(..., mode="fan_in", nonlinearity="linear")`
  for SELU layers; this matches LeCun normal initialization.
- Keep comparisons between ReLU/dropout and SELU/AlphaDropout conceptually clear.
- Do not add BatchNorm to SNN examples unless explicitly requested; it changes the
  comparison.

## Repository Structure

- `TF_2_x/` contains current TensorFlow/Keras scripts.
- `Pytorch/` contains PyTorch notebooks.
- The root `environment.yml` is the main environment for current TF2 and PyTorch
  examples.
