import network_cir
import numpy as np

import torch
import torch.nn as nn


class FFT_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, block_size):
        super(FFT_Linear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_size = block_size
        self.cirmat_size = [int(self.out_channels/self.block_size), int(self.in_channels/self.block_size), block_size]
        print(self.cirmat_size)

        self.cir_weights = nn.Parameter(torch.rand(self.cirmat_size))
    
    def forward(self, x):
        block_size = self.cirmat_size[2]
        x_input = [x[k*block_size:(k+1)*block_size] for k in range(self.cirmat_size[1])] 
        a = [torch.zeros(block_size) for _ in range(self.cirmat_size[0])]

        for i in range(self.cirmat_size[0]):
            for j in range(self.cirmat_size[1]):
                weights = self.cir_weights[i, j]
                weights = torch.stack([weights, torch.zeros(block_size)], dim=0)
                a_fft = torch.fft(torch.Tensor(weights.T), signal_ndim=1)

                x_in = x_input[j]
                x_in = torch.stack([x_in, torch.zeros(block_size)], dim=0)
                b_fft = torch.fft(torch.Tensor(x_in.T), signal_ndim=1)


                ab_r_1 = a_fft[:,0] * b_fft[:,0]
                ab_r_2 = a_fft[:,1] * b_fft[:,1]
                ab = torch.add(a_fft[:,0], a_fft[:,1]) *  torch.add(b_fft[:,0], b_fft[:,1])

                res_r = ab_r_1 - ab_r_2
                res_i = ab - ab_r_1 - ab_r_2
                res = torch.stack([res_r, res_i], axis=1)
                iff_res = torch.ifft(res, signal_ndim=1)

                a[i] += iff_res[:,0]
        
        output = a[0]
        for tensor in a[1:]:
            output = torch.cat((output, tensor), dim=0)
        return output


fftl = FFT_Linear(10, 10, 5)

test_data = torch.rand((10, 1))
print("test data:")
print(test_data)

weights = fftl.cir_weights.detach().numpy()

print("weights0 data 1:")
print(weights[0,0])
print("weights0 data 2:")
print(weights[0,1])

num_layers = 1
cir_matrix_num = fftl.cirmat_size[:2]
print(cir_matrix_num)

def recover_matrix(vec):
    cir_w = np.hstack((vec[0], np.flip(vec[1:])))
    cweights = cir_w.reshape(1, -1)
    for j in range(vec.shape[0]-1):
        cweights = np.vstack((cweights, np.hstack((cir_w[-j-1:], cir_w[:-j-1])).reshape(1,-1)))
    
    return cweights

x = test_data
layer_matr = None
for i in range(cir_matrix_num[0]):
    temp = None
    for j in range(cir_matrix_num[1]):
        matr = recover_matrix(weights[i,j])
        if temp is not None:
            temp = np.hstack([temp, matr])
        else:
            temp = matr
    if layer_matr is not None:
        layer_matr = np.vstack((layer_matr, temp))
    else:
        layer_matr = temp
x = np.dot(layer_matr, x)


print("recovered matrix version:")
print(x)
print("net output:")
test_data = torch.Tensor(test_data.reshape(-1))
print(fftl.forward(test_data))