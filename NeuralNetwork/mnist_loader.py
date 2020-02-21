import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open("../mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, 784) for x in tr_d[0]]
    # print(np.asarray(training_inputs).shape)
    # training_inputs = [x[0:780] for x in training_inputs[0:2000]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = [training_inputs, training_results]

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_inputs = [x[0:780,:] for x in validation_inputs]
    validation_data = zip(validation_inputs, va_d[1])
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_inputs = [x[0:780,:] for x in test_inputs[0:10]]
    # print(np.asarray(test_inputs).shape)
    # test_data = zip(test_inputs, te_d[1])
    test_data = zip(training_inputs, tr_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e
