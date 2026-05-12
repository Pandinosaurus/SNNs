# Self-Normalizing Networks

Tutorials and implementations for Self-Normalizing Networks (SNNs) as suggested
by Klambauer et al. ([arXiv pre-print](https://arxiv.org/pdf/1706.02515.pdf)).

The repository contains TensorFlow/Keras scripts and PyTorch notebooks for 
demonstrating the main SNN recipe:

- SELU activations
- LeCun normal initialization
- AlphaDropout when dropout is used

## Environment

The current examples use TensorFlow 2.x/Keras and PyTorch. Create the Conda
environment from the root environment file:

```bash
conda env create -f environment.yml
conda activate snn
```

The environment is intended for current TensorFlow and PyTorch versions. On
Linux/NVIDIA systems, TensorFlow GPU dependencies are installed via the
`tensorflow[and-cuda]` pip extra.

## Implementation Notes

### TensorFlow/Keras

Keras provides SELU, LeCun normal initialization, and AlphaDropout:

- [`tf.keras.activations.selu`](https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu)
- [`tf.keras.initializers.LecunNormal`](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal)
- [`tf.keras.layers.AlphaDropout`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout)

### PyTorch

PyTorch provides SELU and AlphaDropout:

- [`torch.nn.SELU`](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html)
- [`torch.nn.AlphaDropout`](https://pytorch.org/docs/stable/generated/torch.nn.AlphaDropout.html)

For SELU networks, initialize linear and convolutional layers with:

```python
torch.nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="linear")
```

This corresponds to LeCun normal initialization because `fan_in` is used with a
gain of 1.

## Tutorials

### TensorFlow 2.x / Keras

- Multilayer Perceptron on MNIST ([python script](TF_2_x/MNIST-MLP-SELU.py))
- Convolutional Neural Network on MNIST ([python script](TF_2_x/MNIST-Conv-SELU.py))
- Convolutional Neural Network on CIFAR10 ([python script](TF_2_x/CIFAR10-Conv-SELU.py))

### PyTorch

- Multilayer Perceptron on MNIST ([notebook](Pytorch/SelfNormalizingNetworks_MLP_MNIST.ipynb))
- Convolutional Neural Network on MNIST ([notebook](Pytorch/SelfNormalizingNetworks_CNN_MNIST.ipynb))
- Convolutional Neural Network on CIFAR10 ([notebook](Pytorch/SelfNormalizingNetworks_CNN_CIFAR10.ipynb))

## Further Material

### Figure 1

Notebooks and code to produce Figure 1 are provided in [figure1](figure1/).
This material builds on top of the [biutils](https://github.com/untom/biutils)
package.

### Calculations and Numeric Checks

Mathematica notebooks for calculations and numeric checks of the theorems are
provided here:

- [Mathematica notebook](Calculations/SELU_calculations.nb)
- [Mathematica PDF](Calculations/SELU_calculations.pdf)

### UCI, Tox21, and HTRU2 Data Sets

- [UCI](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz)
- [Tox21](http://bioinf.jku.at/research/DeepTox/tox21.zip)
- [HTRU2](https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip)
