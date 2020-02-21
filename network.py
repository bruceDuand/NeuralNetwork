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
            x = sigmoid(np.dot(w, x))
        return x

    def Adam(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n_train = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] 
                            for k in range(0, n_train, mini_batch_size)]

        # M_bs = [np.zeros(b.shape) for b in self.biases]
        # R_bs = [np.zeros(b.shape) for b in self.biases]
        M_ws = [np.zeros(w.shape) for w in self.weights]
        R_ws = [np.zeros(w.shape) for w in self.weights]
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for j in range(1, epochs+1):
            eta = np.power(1.0000001,-j)*eta

            idx = np.random.randint(0, len(mini_batches))
            mini_batch = mini_batches[idx]

            # for mini_batch in mini_batches:
            #     self.update_mini_batch(mini_batch, eta_dec)

            # self.update_mini_batch(mini_batch, eta, j+1)
            delta_ws, delta_bs = self.get_mini_batch_grad(mini_batch)

            for k in range(self.num_layers-1):
                M_ws[k] = beta1 * M_ws[k] + (1 - beta1) * delta_ws[k]
                R_ws[k] = beta2 * R_ws[k] + (1 - beta2) * delta_ws[k] ** 2

                m_k_hat = M_ws[k] / (1. - beta1**(j))
                r_k_hat = R_ws[k] / (1. - beta2**(j))

                self.weights[k] -= eta * m_k_hat / (np.sqrt(r_k_hat) + eps)

            self.biases = [b-eta*delta_b for b, delta_b in zip(self.biases, delta_bs)]

            if test_data:
                print("Epoch: {}, Accuracy: {} / {}, eta: {}".format(j, self.evaluate(test_data), n_test, eta))
            else:
                print("Epoch: {} completed".format(j))

    def get_mini_batch_grad(self, mini_batch):
        total_delta_bs = [np.zeros(b.shape) for b in self.biases]
        total_delta_ws = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_bs, delta_ws = self.backprop(x, y)
            total_delta_bs = [tb+b for tb, b in zip(total_delta_bs, delta_bs)]
            total_delta_ws = [tw+w for tw, w in zip(total_delta_ws, delta_ws)]
        total_delta_bs = [total_delta_b/len(mini_batch) for total_delta_b in total_delta_bs]
        total_delta_ws = [total_delta_w/len(mini_batch) for total_delta_w in total_delta_ws]

        return total_delta_ws, total_delta_bs


    def update_mini_batch(self, mini_batch, eta, t):
        M_bs = [np.zeros(b.shape) for b in self.biases]
        R_bs = [np.zeros(b.shape) for b in self.biases]
        M_ws = [np.zeros(w.shape) for w in self.weights]
        R_ws = [np.zeros(w.shape) for w in self.weights]
        beta1 = 0.9
        beta2 = 0.999

        # get_minibatch_grad
        total_delta_bs = [np.zeros(b.shape) for b in self.biases]
        total_delta_ws = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_bs, delta_ws = self.backprop(x, y)
            total_delta_bs = [tb+b for tb, b in zip(total_delta_bs, delta_bs)]
            total_delta_ws = [tw+w for tw, w in zip(total_delta_ws, delta_ws)]
        total_delta_bs = [total_delta_b/len(mini_batch) for total_delta_b in total_delta_bs]
        total_delta_ws = [total_delta_w/len(mini_batch) for total_delta_w in total_delta_ws]

        for k in range(self.num_layers-1):
            M_ws[k] = beta1 * M_ws[k] + (1 - beta1) * total_delta_ws[k]
        
        self.weights = [w-eta*total_delta_w for w, total_delta_w in zip(self.weights, total_delta_ws)]
        self.biases = [b-eta*total_delta_b for b, total_delta_b in zip(self.biases, total_delta_bs)]

    def backprop(self, x, y):
        delta_bs = [np.zeros(b.shape) for b in self.biases]
        delta_ws = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)
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

def relu(z):
    res = np.zeros_like(z)
    for i in range(z.shape[0]):
        res[i] = z[i] if z[i] > 0 else 0
    return res

def relu_prime(z):
    res = np.zeros_like(z)
    for i in range(z.shape[0]):
        res[i] = 1 if z[i] > 0 else 0
    return res
