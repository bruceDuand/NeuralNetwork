import network_cir
import numpy as np

net = network_cir.Network(sizes=[780, 30, 10], cir_size=[10, 5])
x = np.random.randn(780, 1)
y = net.feedforward(x)
