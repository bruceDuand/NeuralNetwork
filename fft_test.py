import numpy as np

cir_W = np.random.randint(3, size=(1, 10)).tolist()[0]
# print(cir_W)
cir_matr = np.asarray(cir_W)

l = len(cir_W)
for i in range(l-1):
    cir_matr = np.vstack((cir_matr, np.hstack((cir_W[-i-1:], cir_W[:-i-1]))))

print(cir_matr)

input_data = np.random.randint(3, size=(10, 1))
print("input data:")
print(input_data.T)

res = np.dot(cir_matr, input_data)
print("normal MVC result:")
print(res.T)

print("Using FFT result:")
cir_fft = np.fft.fft(cir_matr[:,0])
print(cir_fft.shape)

input_fft = np.fft.fft(input_data.T)
print(input_fft.shape)

res_fft = cir_fft * input_fft
res = np.fft.ifft(res_fft)
print(np.abs(res))

