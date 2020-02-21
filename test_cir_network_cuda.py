import network_cir_cuda
import numpy as np
from mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

net = network_cir_cuda.Network(sizes=[780, 60, 10], cir_size=[30, 10])
net.SGD(training_data, 500, 20, 1e-2, test_data=test_data)
