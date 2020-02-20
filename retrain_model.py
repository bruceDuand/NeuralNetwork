import sys
sys.path.append('../')
import network_cir
from preprocessings import load_from_pickle, save_to_pickle
import numpy as np

x_train_fc_in = load_from_pickle(filename="fc-train-in.pkl")
x_test_fc_in = load_from_pickle(filename="fc-test-in.pkl")
y_train = load_from_pickle(filename="y-train.pkl")
y_test = load_from_pickle(filename="y-test.pkl")

x_train_fc_in = [np.reshape(x, (-1, 1)) for x in x_train_fc_in]
x_test_fc_in = [np.reshape(x, (-1, 1)) for x in x_test_fc_in]
y_train = [np.reshape(x, (-1, 1)) for x in y_train]
y_test = [np.reshape(x, (-1, 1)) for x in y_test]

test_data = zip(x_test_fc_in, y_test)
training_data = zip(x_train_fc_in, y_train)


net = network_cir.Network(sizes=[80, 32, 2], cir_size=[8, 2])
net.SGD(training_data, 100, 100, 3.0, test_data=test_data)


cir_weights = net.cir_weights
cir_biases = net.biases
save_to_pickle(cir_weights, "weights.pkl")
save_to_pickle(cir_biases, "biases.pkl")
# original trained weights and biasas are in files named cir_weights and cir_biases