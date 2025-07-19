# NeuralNetworks Python library
# by egnotech

import numpy as np
import os, sys, pickle

N_INPUTS = 1
N_OUTPUTS = 2
HIDDEN_LAYERS = [3, 5, 5, 4]
BATCH_SIZE = 10

def inp():
    tL = input("Start training or load previous? (train/load): ")
    network = None

    if tL == "train":
        network = NeuralNet(N_INPUTS, N_OUTPUTS, HIDDEN_LAYERS, BATCH_SIZE)
        network.training_data()
    elif tL == "load":
        try:
            file = input("Please choose a file (leave blank for default: network.pkl). ")
            if not file:
                file = "network.pkl"
            network = NeuralNet(N_INPUTS, N_OUTPUTS, HIDDEN_LAYERS, BATCH_SIZE)
            network.load(file)
        except FileNotFoundError:
            print("Error: file not found. Please try again.")
    if network is not None:
        if input("The network is trained. Would you like to test it (y/n)? ") == "y":
            print("Testing neural network. Input \".cancel\" to cancel.")
            while True:
                try:
                    test_input = [float(input(f"NeuralNet: Input {i+1}: ")) for i in range(len(network.activations[0]))]
                    prediction, confidence = network.predict(np.array(test_input))
                    print(f"Network tested succesfully.\nPrediction: {prediction}\nConfidence: {confidence * 100:.2f}%.")
                    if input("Would you like to test again (y/n)? ") == "n":
                        if input("Would you like to save this network (y/n)? ") == "y":
                            save_as = input("Save as (leave blank for default network.pkl, must end in .pkl): ")
                            if not save_as:
                                save_as = "network.pkl"
                            network.save(save_as)
                        break
                except ValueError as err:
                    if ".cancel" in str(err):
                        print("Cancelling...")
                        return
                    print("Input is not valid. Please try again.")

class NeuralNet:
    def __init__(self, n_inputs, n_outputs, hidden_layers, batch_size, verbose=True):
        self.program_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(os.path.join(self.program_dir, "Saved networks"), exist_ok=True)

        self.verbose = verbose
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

        self.layer_sizes = [self.n_inputs] + self.hidden_layers + [self.n_outputs]

        if self.verbose:
            print("NeuralNet: Initializing weights and biases. This might take a while for larger models.")
        self.init_weights_biases()
        if self.verbose:
            print("NeuralNet: Weights and biases initialized. Initializing activations.")
        self.init_activations()
        if self.verbose:
            print("NeuralNet: Done!")
    
    def init_weights_biases(self):
        rng = np.random.default_rng()
        self.weights = [rng.standard_normal((self.layer_sizes[i+1], self.layer_sizes[i]), dtype=np.float32) * np.sqrt(2 / (self.layer_sizes[i] if i < len(self.layer_sizes) - 2 else self.layer_sizes[i] + self.layer_sizes[i+1])) for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.zeros_like(w[:, :1]) for w in self.weights]
    
    def init_activations(self):
        self.activations = [np.zeros((length, self.batch_size)) for length in self.layer_sizes]
        self.z_values = [np.zeros((length, self.batch_size)) for length in self.layer_sizes]
    
    def training_data(self, epochs=500, learning_rate=0.05):
        if self.verbose:
            print("NeuralNet: Input training data. Please input \".cancel\" at any point to stop the process.")
        while True:
            try:
                results = [
                    np.array([[float(input(f"NeuralNet: Input {i+1}, batch {j+1}: ")) for i in range(len(self.activations[0]))] for j in range(self.batch_size)]).T,
                    np.array([[float(input(f"NeuralNet: Label output {i+1}, batch {j+1}: ")) for i in range(len(self.activations[-1]))] for j in range(self.batch_size)]).T
                ]
                if self.verbose:
                    print("NeuralNet: Training neural network. This might take a while.")
                self.train_epochs(results, epochs, learning_rate)
                return
            except ValueError as err:
                if ".cancel" in str(err):
                    if self.verbose:
                        print("NeuralNet: Cancelling...")
                    return
                print("NeuralNet: Input is not valid. Please try again.")
    
    def softmax(self, n):
        exp_n = np.exp(n - np.max(n, axis=0, keepdims=True))
        return exp_n / np.sum(exp_n, axis=0, keepdims=True)
    
    def leaky_relu(self, n):
        return np.maximum(0.1 * n, n)
    
    def propagate(self):
        for i in range(len(self.weights) - 1):
            self.z_values[i + 1] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.activations[i + 1] = self.leaky_relu(self.z_values[i + 1])
        self.activations[-1] = self.softmax(np.dot(self.weights[-1], self.activations[-2]) + self.biases[-1])
    
    def loss(self, true, pred):
        return -np.mean(np.sum(true * np.log(np.clip(pred, 1e-12, 1.0)), axis=1))
    
    def acc(self, true, pred):
        predictions = np.argmax(pred, axis=0)
        targets = np.argmax(true, axis=0)
        return np.mean(predictions == targets)
    
    def train(self, results, learning_rate=0.05):
        inputs, targets = results
        self.activations[0] = inputs
        self.propagate()

        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        delta = self.activations[-1] - targets

        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            d_weights[i] = np.dot(delta, a_prev.T) / self.batch_size
            d_biases[i] = np.mean(delta, axis=1, keepdims=True)

            if i != 0:
                z = self.z_values[i]
                d_lrelu = np.where(z > 0, 1.0, 0.1)
                delta = np.dot(self.weights[i].T, delta) * d_lrelu
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * d_weights[i]
            self.biases[i] -= learning_rate * d_biases[i]
    
    def train_epochs(self, results, epochs=500, learning_rate=0.05):
        inputs, targets = results
        samples = inputs.shape[1]

        for epoch in range(epochs):
            indices = np.random.permutation(samples)
            shuff_inputs = inputs[:, indices]
            shuff_targets = targets[:, indices]

            self.train((shuff_inputs, shuff_targets), learning_rate)
            loss = self.loss(shuff_targets, self.activations[-1])
            acc = self.acc(shuff_targets, self.activations[-1])
            if self.verbose:
                print(f"Epoch {epoch} - Loss: {loss:.5f}; Accuracy: {acc * 100:.2f}%")

    def predict(self, inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        
        self.activations[0] = inputs
        self.propagate()

        output = self.activations[-1]
        prediction = np.argmax(output, axis=0)
        confidence = output[prediction, np.arange(output.shape[1])]

        return int(prediction[0]), float(confidence[0])
    
    def save(self, filename="network.pkl"):
        base = filename[:-4]
        new = base

        i = 0
        while os.path.exists(os.path.join(self.program_dir, "Saved networks", f"{new}.pkl")):
            i += 1
            new = f"{base}_{i}"
        filename = os.path.join(self.program_dir, "Saved networks", f"{new}.pkl")

        with open(filename, "wb") as file:
            pickle.dump({
                "Inputs": self.n_inputs,
                "Outputs": self.n_outputs,
                "Hidden layers": self.hidden_layers,
                "Batch size": self.batch_size,
                "Weights": self.weights,
                "Biases": self.biases
            }, file)
        if self.verbose:
            print(f"NeuralNet: Saved to '{filename}'.")
    
    def load(self, filename="network.pkl"):
        with open(os.path.join(self.program_dir, "Saved networks", filename), "rb") as file:
            data = pickle.load(file)
        self.n_inputs = data["Inputs"]
        self.n_outputs = data["Outputs"]
        self.hidden_layers = data["Hidden layers"]
        self.batch_size = data["Batch size"]
        self.layer_sizes = [self.n_inputs] + self.hidden_layers + [self.n_outputs]
        self.weights = data["Weights"]
        self.biases = data["Biases"]
        self.init_activations()
        if self.verbose:
            print(f"NeuralNet: Loaded from '{filename}'.")

if __name__ == "__main__":
    inp()
