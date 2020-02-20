import network_cir
import numpy as np
from mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()
net = network_cir.Network(sizes=[780, 30, 10], cir_size=[10, 5])
net.SGD(training_data, 100, 50, 3.0, test_data=test_data)

# print("maxinput x:", np.max(test_x))




# test correctness, feedforward network
# ==================================
# net = network_cir.Network(sizes=[4, 4, 2], cir_size=[2, 2])
# test_data = np.random.randint(3, size=(4, 1))
# print("test data:")
# print(test_data)

# weights = net.cir_weights
# print("weights0 data 1:")
# print(weights[0][0,0])
# print("weights0 data 2:")
# print(weights[0][0,1])
# print("weights0 data 3:")
# print(weights[0][1,0])
# print("weights0 data 4:")
# print(weights[0][1,1])


# print("weights1 data 1:")
# print(weights[1][0,0])
# print("weights1 data 2:")
# print(weights[1][0,1])


# biases = net.biases
# num_layers = net.num_layers
# cir_matrix_num = net.cir_matrix_num




# def recover_matrix(vec):
#     cir_w = np.hstack((vec[0, 0], np.flip(vec[0, 1:])))
#     cweights = cir_w.reshape(1, -1)
#     for j in range(vec.shape[1]-1):
#         cweights = np.vstack((cweights, np.hstack((cir_w[-j-1:], cir_w[:-j-1])).reshape(1,-1)))
    
#     return cweights

# x = test_data
# for layer_idx in range(num_layers-1):
#     layer_matr = None
#     for i in range(cir_matrix_num[layer_idx][0]):
#         temp = None
#         for j in range(cir_matrix_num[layer_idx][1]):
#             matr = recover_matrix(weights[layer_idx][i,j])
#             if temp is not None:
#                 temp = np.hstack([temp, matr])
#             else:
#                 temp = matr
#         if layer_matr is not None:
#             layer_matr = np.vstack((layer_matr, temp))
#         else:
#             layer_matr = temp
#     print(layer_matr)
#     x = np.dot(layer_matr, x) + biases[layer_idx]


# print("recovered matrix version:")
# print(x)
# print("net output:")
# print(net.feedforward(test_data))
        
        