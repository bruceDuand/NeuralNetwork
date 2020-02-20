import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n_train = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                                for k in range(0, n_train, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch: {} : Accuracy: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch: {} completed".format(j))

    def update_mini_batch(self, mini_batch, eta):
        total_delta_bs = [np.zeros(b.shape) for b in self.biases]
        total_delta_ws = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_bs, delta_ws = self.backprop(x, y)
            total_delta_bs = [tb+b for tb, b in zip(total_delta_bs, delta_bs)]
            total_delta_ws = [tw+w for tw, w in zip(total_delta_ws, delta_ws)]
        
        self.weights = [w-eta*total_delta_w/len(mini_batch) for w, total_delta_w in zip(self.weights, total_delta_ws)]
        self.biases = [b-eta*total_delta_b/len(mini_batch) for b, total_delta_b in zip(self.biases, total_delta_bs)]

    def backprop(self, x, y):
        delta_bs = [np.zeros(b.shape) for b in self.biases]
        delta_ws = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        delta_bs[-1] = delta
        delta_ws[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_bs[-l] = delta
            delta_ws[-l] = np.dot(delta, activations[-l-1].transpose())

        return (delta_bs, delta_ws)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) 
                            for (x, y) in test_data]
        return sum([int(x==y) for x, y in test_results])

    def cost_derivative(self, output_activations, target):
        return (output_activations - target)

    
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))