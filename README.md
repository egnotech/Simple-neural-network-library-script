# NeuralNets library
This is a simple yet versatile neural network Python script, which can also be imported as a library. It has a verbose mode and non-verbose mode.

Here is an example of how to use the library.

First, you must import the module:
```python
# Import the module
from NeuralNetworks import NeuralNet
```
Now, create a NeuralNet object to initialize the neural network. NeuralNet() has the following syntax: 
```python
NeuralNetworks.NeuralNet(n_inputs, n_outputs, hidden_layers, batch_size, verbose=True)
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

Next, if you want to train your neural network, you need to code an input system. This will supply the training data. If you do not want to do this, then you can instead import the basic built-in input system from the module. You can do this in two ways:
1. Add this line to your code:
   ```python
   from NeuralNetworks import inp
   ```
   Then, simply calling the `inp()` function will start your input system, automatically call all required functions, and automatic train your neural network in epochs.
2. Or, you can import the whole module at once, by replacing
   ```python
   from NeuralNetworks import NeuralNet
   ```
   with
   ```python
   import NeuralNetworks
   ```
   Then, you must replace
   ```python
   network = NeuralNet(INPUTS, OUTPUTS, HIDDEN_LAYERS, BATCH_SIZE)
   ```
   with
   ```python
   network = NeuralNetworks.NeuralNet(INPUTS, OUTPUTS, HIDDEN_LAYERS, BATCH_SIZE)
   ```
   Once again, calling `NeuralNetworks.inp()` will start you input system and everything else automatically.

These will both give the same output (assuming verbose mode is on):
```
Start training or load previous? (train/load): 
```
However, if you don't want to use the built-in function for inputs, here is some helpful information for coding your own:

There is a built in `NeuralNetworks.NeuralNet.training_data()` function which will be mostly silenced with verbose mode. This can be a shortcut for your input system. It will automatically ask for training data input, and then train the neural network correspondingly.

If you want to do this manually, though, please note that the output of your input function must have the shape: `[np.array(batch_size, n_inputs), np.array(batch_size, n_outputs)]`. Then, you must pass this output as the first argument of the `NeuralNetworks.NeuralNet.train_epochs()` function. It has the following syntax:
```python
NeuralNetworks.NeuralNet.train_epochs(results, epochs=500, learning_rate=0.05)
```
Where `results` is the argument mentioned above, `epochs` is an optional argument defaulting to 500 which specifies the number of epochs, and `learning_rate` is another optional argument, defaulting to 0.05, specifying the learning rate. This will train the neural network.

Here are some other useful functions built-in:
```python
NeuralNetworks.NeuralNet.predict(inputs)
```
`inputs` is an array (supports batch processing) with the inputs to the neural network. Shape: `np.array(batch_size, n_inputs)` or `np.array(n_inputs)`
```python
NeuralNetworks.NeuralNet.loss(true, pred)
```
loss between the correct (true) values `true` and the neural network's predicted values `pred` (both with shape `np.array(batch_size, n_outputs)`).
```python
NeuralNetworks.NeuralNet.acc(true, pred)
```
same as the `loss()` function above but it calculates the accuracy.
```python
NeuralNetworks.NeuralNet.
```
```python
NeuralNetworks.NeuralNet.
```
```python
NeuralNetworks.NeuralNet.
```
