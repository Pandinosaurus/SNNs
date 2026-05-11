---
name: self-normalizing-networks
description: Use this skill when working in the Self-Normalizing Networks tutorial repository, especially for TensorFlow, PyTorch, Conda environment, or SELU/AlphaDropout implementation tasks.
---

# Self-Normalizing Networks

This repository contains tutorial implementations for Self-Normalizing Networks
(SNNs) based on Klambauer et al. Preserve the teaching purpose of the examples:
show how SELU networks are constructed and compared across TensorFlow and
PyTorch implementations.

## Core SNN Rules

- Pair SELU activations with LeCun normal initialization.
- Use AlphaDropout when dropout is used in SELU networks.
- In PyTorch, use `nn.init.kaiming_normal_(..., mode="fan_in", nonlinearity="linear")`
  for SELU layers; this matches LeCun normal initialization.
- Keep comparisons between ReLU/dropout and SELU/AlphaDropout conceptually clear.
- Do not add BatchNorm to SNN examples unless explicitly requested; it changes the
  comparison.

## Repository Structure

- `TF_1_x/` contains legacy TensorFlow 1.x material.
- `TF_2_x/` contains current TensorFlow/Keras scripts.
- `Pytorch/` contains PyTorch notebooks.
- The root `environment.yml` is the main environment for current TF2 and PyTorch
  examples. `TF_1_x/environment.yml` belongs to the legacy TF1 material.

## Maintenance Guidance

- Treat `TF_1_x/` as legacy unless a task explicitly asks to update it.
- Prefer current `tf.keras`/Keras APIs in `TF_2_x/`.
- Prefer idiomatic PyTorch training/evaluation loops in `Pytorch/`.
- Keep dataset splits clean: train on training data, validate on a validation split,
  and reserve the test set for final evaluation.
- Avoid saving trained models or generated result files unless they are used by
  another script or explicitly requested.

## Environment Guidance

- Keep the environment file concise and portable; do not commit raw machine-specific
  `conda env export` output.
- Do not reintroduce old CUDA toolkit pins such as `cudatoolkit=10.1`.
- For Linux/NVIDIA systems, TensorFlow GPU dependencies should use the modern
  `tensorflow[and-cuda]` pip extra.

## Validation

- For Python scripts, run a lightweight smoke test before finishing when possible.
  Reducing `epochs` to `0` is acceptable for import/build/evaluate checks.
- For notebooks, prefer small-subset or quick cell-level tests when full training is
  too expensive.
- Verify imports for TensorFlow, PyTorch, torchvision, NumPy, scikit-learn,
  matplotlib, and tqdm after environment changes.
