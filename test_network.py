import network
import numpy as np
from mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)

net = network.Network([780, 30, 10])
net.Adam(training_data, 500, 20, 1e-3, test_data=test_data)

# def vectorized_result(j):
#     e = np.zeros((4, 1))
#     e[j] = 1.0
#     return e

# num_classes = 4
# test_x = np.random.randn(100,20,1)
# test_y = np.random.randint(num_classes-1, size=(100, 1))

# test_y_onehot = [vectorized_result(y) for y in test_y]
# training_data = zip(test_x, test_y_onehot)
# test_data = zip(test_x, test_y)

# net = network.Network([20, 10, 4])
# for i in range(20):
#     print(net.feedforward(test_x[i]))

# net.SGD(training_data, 300, 20, 0.5, test_data=test_data)