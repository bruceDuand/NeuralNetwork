import torch
import torch.nn as nn
from mnist_loader import load_data_wrapper
from torch.autograd import Variable

import numpy as np

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(in_features=784, out_features=30, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=10, bias=True)
        )
    
    def forward(self, x):
        return self.FC(x)

training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)

# print(np.asarray(training_data[0]).shape)
x_train_tensor = torch.Tensor(training_data[0])
y_train_tensor = torch.Tensor(training_data[1])
print(y_train_tensor.size())

net = Network()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
loss_function = nn.MSELoss()


for cur_iter in range(500):
    print("iter: {:2d}".format(cur_iter+1), end=", ")
    outputs = net(x_train_tensor)
    optimizer.zero_grad()
    loss = loss_function(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()


    _, outputs_label = outputs.max(dim=1)
    _, target_label = y_train_tensor.max(dim=1)
    # print(outputs_label)
    # print(target_label)
    accuracy = int(sum(outputs_label == target_label))/len(target_label)
    print("accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss))
