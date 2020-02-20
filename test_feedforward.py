import network_cir
import numpy as np


class FFT_forward():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.matrix_size = [int(p/n) for p, n in zip(sizes[:-1], sizes[1:])]
        # print(self.matrix_size) # [26, 3]

        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.matrix_size, sizes[1:])]

    def fft_feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            l = w.shape[1] # 30, 10
            # print(l)
            tmp = np.zeros_like(b)
            for cir_id in range(w.shape[0]):
                print(w[cir_id, :])
                tmp = tmp + np.fft.ifft(np.fft.fft(w[cir_id, :].T) * np.fft.fft(x[cir_id*l:(cir_id+1)*l]))
            # print(tmp.shape)
            x = sigmoid(np.real(tmp + b)
            # print(x.shape)
        return np.real(x)

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


net = network_cir.Network([780, 30, 10])
x = np.random.randn(780)
print("input x:")
print(x)


fres = net.fft_feedforward(x)
print("fft output:")
print(fres)

weights = net.weights
bias = net.biases

all_weights = []
# print(weights[1][0,:])
print("#######################")
for weight_matr in weights:
    reco_cir_matr = None
    for line_idx in range(weight_matr.shape[0]):
        cir_w = weight_matr[line_idx, :]
        # print(cir_w)
        # print(type(cir_w))
        cir_w = np.hstack((cir_w[0], np.flip(cir_w[1:])))
        # print(cir_w)
        weight = np.array(cir_w)
        for i in range(weight_matr.shape[1]-1):
            weight = np.vstack((weight, np.hstack((cir_w[-i-1:], cir_w[:-i-1]))))
        if line_idx == 0:
            reco_cir_matr = weight
        else:
            reco_cir_matr = np.hstack((reco_cir_matr, weight))
    all_weights.append(reco_cir_matr)

print(all_weights[0])

for i in range(2):
    x = sigmoid(np.dot(all_weights[i], x) + bias[i])

print(x)
    


# weights_0 = np.array(weights[0][0,:])
# for i in range(1, weights[0].shape[0]):
#     weights_0 = np.hstack((weights_0, weights[0][i,:]))

# print(weights_0.shape)

# l = len(weights[0].shape[0])
# for i in range(l-1):
#     cir_matr = np.vstack((cir_matr, np.hstack((cir_W[-i-1:], cir_W[:-i-1]))))

# [-1.19245939  0.88053802  0.50532383 -0.77019129 -0.51387145  2.63127398
#   0.29616636  0.42703008  1.16026774  0.38030688]