# Simple-neural-network-library-script
This is a simple yet versatile neural network Python script, which can also be imported as a library. It has a verbose mode and non-verbose mode.

Here is an example of how to use the library.

First, you must import the module:
```python
# Import the module
from NeuralNetworks import NeuralNet
```
Now, create a NeuralNet object to initialize the neural network. NeuralNet() has the following syntax: 
```python
NeuralNet(n_inputs, n_outputs, hidden_layers, batch_size, verbose=True)
```
It creates a neural network with `n_inputs` inputs and `n_outputs` outputs. `hidden_layers` is a list containing the number of neurons in each hidden layer. It will be trained with a batch size of `batch_size` (batch gradient descent, batch size = size of dataset). Finally, there is an optional `verbose` flag which defaults to True, setting this as False will disable all non-necessary console outputs from the NeuralNet class.
```python
# Create a NeuralNet object to initialize the neural network.
network = NeuralNet(INPUTS, OUTPUTS, HIDDEN_LAYERS, BATCH_SIZE)
```
With verbose mode, the console output will be:
```
NeuralNet: Initializing weights and biases. This might take a while for larger models.
NeuralNet: Weights and biases initialized. Initializing activations.
NeuralNet: Done!
```
Otherwise, there will be no output.

Next, if you want to train your neural network, you need to code an input system. This will supply the training data. If you do not want to do this, then you can instead import in the basic input system built-in to the module.
