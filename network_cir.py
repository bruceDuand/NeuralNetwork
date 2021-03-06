import random
import numpy as np

import torch

class Network(object):
    # ok
    def __init__(self, sizes, cir_size):
        self.num_layers = len(sizes)
        self.sizes = sizes # [780, 30, 10]

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.cir_size = cir_size # [30, 10]

        self.heights = [matr.shape[0] for matr in self.weights]
        self.widths  = [matr.shape[1] for matr in self.weights]
        self.cir_matrix_num = [[int(h/s), int(w/s)] for h, w, s in zip(self.heights, self.widths, self.cir_size)]
        print(self.cir_matrix_num) # [[1, 26], [1, 3]]

        self.cir_weights = [np.random.randn(x[0], x[1], 1, size) for x, size in zip(self.cir_matrix_num, self.cir_size)]
        # print(np.asarray(self.cir_weights[0]).shape)
        # print(np.asarray(self.cir_weights[1]).shape)
        print("#######################")

    # ok
    def feedforward(self, x):
        for layer_idx in range(self.num_layers-1):
            block_size = self.cir_size[layer_idx]
            # print(block_size)
            # print(np.asarray(x).shape)
            x_input = [np.asarray(x[k*block_size:(k+1)*block_size]).reshape(-1, 1) for k in range(self.cir_matrix_num[layer_idx][1])]

            # print(layer_idx)
            # print("x_input_shape:", np.asarray(x_input).shape)
            a = [np.zeros((self.cir_size[layer_idx], 1)) for _ in range(self.cir_matrix_num[layer_idx][0])]
            # print(np.asarray(a).shape)
            for i in range(self.cir_matrix_num[layer_idx][0]):
                for j in range(self.cir_matrix_num[layer_idx][1]):
                    a[i] += getFFTout(self.cir_weights[layer_idx][i, j].T, x_input[j])
            x = np.vstack(a)
            # x = sigmoid(x)

        # print(x)
        return x

    def feedforward_fft(self, x):
        cir_weights = self.cir_weights
        cir_weights = [torch.tensor(weight) for weight in cir_weights]

        for layer_idx in range(self.num_layers-1):
            block_size = self.cir_size[layer_idx]
            # print(block_size)
            # print(np.asarray(x).shape)
            x_input = [np.asarray(x[k*block_size:(k+1)*block_size]).reshape(-1, 1) for k in range(self.cir_matrix_num[layer_idx][1])]

            x_input = [torch.tensor(x) for x in x_input]

            # print(layer_idx)
            # print("x_input_shape:", np.asarray(x_input).shape)
            a = [np.zeros((self.cir_size[layer_idx], 1)) for _ in range(self.cir_matrix_num[layer_idx][0])]
            # print(np.asarray(a).shape)
            for i in range(self.cir_matrix_num[layer_idx][0]):
                for j in range(self.cir_matrix_num[layer_idx][1]):
                    a[i] += getFFTout_torch(cir_weights[layer_idx][i, j].T, x_input[j])
            x = np.vstack(a)
            # x = sigmoid(x)

        # print(x)
        return x

    # ok
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n_train = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            eta_dec = np.power(1.01,-j)*eta


            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                                for k in range(0, n_train, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta_dec)

            if test_data:
                print("Epoch: {}, Accuracy: {} / {}, eta:{}".format(j+1, self.evaluate(test_data), n_test, eta_dec))
            else:
                print("Epoch: {} completed".format(j+1))


    # def Adam(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    #     training_data = list(training_data)
    #     n_train = len(training_data)

    #     if test_data:
    #         test_data = list(test_data)
    #         n_test = len(test_data)

    #     random.shuffle(training_data)
    #     mini_batches = [training_data[k:k+mini_batch_size] 
    #                         for k in range(0, n_train, mini_batch_size)]

    #     # M_bs = [np.zeros(b.shape) for b in self.biases]
    #     # R_bs = [np.zeros(b.shape) for b in self.biases]
    #     M_ws = [np.zeros(w.shape) for w in self.cir_weights]
    #     R_ws = [np.zeros(w.shape) for w in self.cir_weights]
    #     beta1 = 0.9
    #     beta2 = 0.999
    #     eps = 1e-8

    #     for j in range(1, epochs+1):
    #         eta = np.power(1.0000001,-j)*eta

    #         idx = np.random.randint(0, len(mini_batches))
    #         mini_batch = mini_batches[idx]

    #         # for mini_batch in mini_batches:
    #         #     self.update_mini_batch(mini_batch, eta_dec)

    #         # self.update_mini_batch(mini_batch, eta, j+1)
    #         delta_ws, delta_bs = self.get_mini_batch_grad(mini_batch)

    #         for k in range(self.num_layers-1):
    #             M_ws[k] = beta1 * M_ws[k] + (1 - beta1) * delta_ws[k]
    #             R_ws[k] = beta2 * R_ws[k] + (1 - beta2) * delta_ws[k] ** 2

    #             m_k_hat = M_ws[k] / (1. - beta1**(j))
    #             r_k_hat = R_ws[k] / (1. - beta2**(j))

    #             self.cir_weights[k] -= eta * m_k_hat / (np.sqrt(r_k_hat) + eps)

    #         # self.biases = [b-eta*delta_b for b, delta_b in zip(self.biases, delta_bs)]

    #         if test_data:
    #             print("Epoch: {}, Accuracy: {} / {}, eta: {}".format(j, self.evaluate(test_data), n_test, eta))
    #         else:
    #             print("Epoch: {} completed".format(j))

    # def get_mini_batch_grad(self, mini_batch):
    #     total_delta_bs = [np.zeros(b.shape) for b in self.biases]
    #     total_delta_ws = [np.zeros(w.shape) for w in self.cir_weights]
    #     for x, y in mini_batch:
    #         delta_bs, delta_ws = self.backprop(x, y)
    #         total_delta_bs = [tb+b for tb, b in zip(total_delta_bs, delta_bs)]
    #         total_delta_ws = [tw+w for tw, w in zip(total_delta_ws, delta_ws)]
    #     total_delta_bs = [total_delta_b/len(mini_batch) for total_delta_b in total_delta_bs]
    #     total_delta_ws = [total_delta_w/len(mini_batch) for total_delta_w in total_delta_ws]

    #     return total_delta_ws, total_delta_bs


    # ok
    def update_mini_batch(self, mini_batch, eta):
        total_delta_bs = [np.zeros(b.shape) for b in self.biases]
        total_delta_ws = [np.zeros(w.shape) for w in self.cir_weights]
        
        for x, y in mini_batch:
            delta_bs, delta_ws = self.backprop(x, y)
            # print(delta_ws[0].shape)
            total_delta_bs = [tb+b for tb, b in zip(total_delta_bs, delta_bs)]
            total_delta_ws = [tw+w for tw, w in zip(total_delta_ws, delta_ws)]
            
        # print("before update:", self.cir_weights[0][0,0])
        # print(total_delta_ws[0][0,0]/len(mini_batch))
        self.biases = [b-eta*total_delta_b/len(mini_batch) for b, total_delta_b in zip(self.biases, total_delta_bs)]
        self.cir_weights = [w-eta*total_delta_w/len(mini_batch) for w, total_delta_w in zip(self.cir_weights, total_delta_ws)]
        # print("after update:", self.cir_weights[0][0,0])

    def backprop(self, x, y):
        delta_bs = [np.zeros(b.shape) for b in self.biases]
        delta_ws = [np.zeros(w.shape) for w in self.cir_weights]
        # print(delta_ws[0].shape)

        # forward
        activation = x
        # print("input x:", x)
        activations = [x]
        zs = []
        for layer_idx in range(self.num_layers-1):
            block_size = self.cir_size[layer_idx]
            x_input = [np.asarray(activation[k*block_size:(k+1)*block_size]).reshape(-1, 1) for k in range(self.cir_matrix_num[layer_idx][1])]

            a = [np.zeros((self.cir_size[layer_idx], 1)) for _ in range(self.cir_matrix_num[layer_idx][0])]
            for i in range(self.cir_matrix_num[layer_idx][0]):
                for j in range(self.cir_matrix_num[layer_idx][1]):
                    # print("cir_weights:", self.cir_weights[layer_idx][i, j].T)
                    a[i] += getFFTout(self.cir_weights[layer_idx][i, j].T, x_input[j])
            z = np.vstack(a)
            zs.append(z)
            # print(z)
            activation = sigmoid(z)
            # activation = z
            activations.append(activation)

        # print("activations[-1]", activations[-1])
        # print("activation:", self.feedforward(x))

        # backprop
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # delta = self.cost_derivative(activations[-1], y)
        delta_bs[-1] = delta
        
        block_size = self.cir_size[-1]
        delta = [np.asarray(delta[k*block_size:(k+1)*block_size]).reshape(-1, 1) for k in range(self.cir_matrix_num[-1][0])]
        for i in range(self.cir_matrix_num[-1][0]):
            for j in range(self.cir_matrix_num[-1][1]):   
                delta_ws[-1][i,j] = getFFTout(delta[i], activations[-2][j*block_size:(j+1)*block_size]).T
                # print(delta_ws[-1][i,j])

        
        # print("delta 1:", delta[0])
        
        for layer_idx in range(2, self.num_layers):
            z = zs[-layer_idx]
            # sp = sigmoid_prime(z)
            sp = sigmoid_prime(z)
            # sp = 1

            block_size = self.cir_size[-layer_idx+1]
            delta_x = [np.zeros((block_size, 1)) for _ in range(self.cir_matrix_num[-layer_idx+1][1])]
            for i in range(self.cir_matrix_num[-layer_idx+1][0]):
                for j in range(self.cir_matrix_num[-layer_idx+1][1]):
                    delta_x[j] += getFFTout(delta[i], self.cir_weights[-layer_idx+1][i, j].T)
            delta = np.vstack(delta_x) * sp
            delta_bs[-layer_idx] = delta

            block_size = self.cir_size[-layer_idx]
            delta = [np.asarray(delta[k*block_size:(k+1)*block_size]).reshape(-1, 1) for k in range(self.cir_matrix_num[-layer_idx][0])]
            # print("delta 2 after:", delta[0])
            for i in range(self.cir_matrix_num[-layer_idx][0]):
                for j in range(self.cir_matrix_num[-layer_idx][1]):                 
                    delta_ws[-layer_idx][i,j] = getFFTout(delta[i], activations[-layer_idx-1][j*block_size:(j+1)*block_size]).T
        return (delta_bs, delta_ws)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) 
                            for (x, y) in test_data]
        # for (x, y) in test_data:
        #     print("target val:", y)
        #     print(self.feedforward(x))
        #     break
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



def getFFTout(x, y):
    x = [a for b in x for a in b] # change x & y into a single array/list, not a nx1 array
    y = [a for b in y for a in b] # otherwise it will do fft on each row of the array
    res = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).reshape(-1, 1)

    # print("input 1 max:", np.amax(x))
    # print("input 2 max:", np.amax(y))

    # test_res = [a for b in res for a in b]
    # max_val = 0.0
    # for i in test_res:
    #     max_val = np.amax([max_val, i])
    # print("output max:", max_val)
    
    # print(np.imag(res))
    return np.real(res)

def getFFTout_torch(x, y):
    return torch.ifft(torch.fft(x), torch.fft(y))
