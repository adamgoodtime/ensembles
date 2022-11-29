import torch
import numpy as np
import time
from numba import jit, njit, cuda, vectorize

"""
pass in data array
    pass in input
        activate neurons
            activate synapses

variables to control:
sparsity of neuron connections
input size
population size
NUMBER OF INPUTS PASSED SIMULTANEOUSLY
"""

def generate_data(sam_size, pop_size, inp_size, spa_city):
    param_matrix = np.random.random((pop_size, inp_size)).astype(np.float32)
    sparsity_mask = np.random.random((pop_size, inp_size)) < spa_city
    sparse_matrix = param_matrix * sparsity_mask
    inputs = np.random.random((sam_size, inp_size)).astype(np.float32)
    return param_matrix, sparsity_mask, inputs

def launch_test(param_m, sparse_m, input_m, test):
    timings = {}
    if test == 'torch':
        timings['torch'] = torch_test(param_m, sparse_m, input_m)
    return timings

def torch_test(params, mask, inputs, gpu=True, autograd=False):
    torch.set_grad_enabled(autograd)
    if gpu:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    w = torch.Tensor(width)
    p = torch.Tensor(params)
    m = torch.Tensor(mask)
    i = torch.Tensor(inputs)

    t0 = time.time_ns()
    sub = torch.subtract(i, p)
    scaled_sub = torch.divide(sub, w)
    diff = torch.abs(scaled_sub)
    diff = torch.multiply(diff, m)
    act = torch.sum(diff, dim=0)
    t1 = time.time_ns()
    print(act[0])
    return t1 - t0




# sparsity = [0.1, 0.33, 0.66, 1]
# number_of_inputs = [1, 10, 100, 1000]
# number_of_samples = [1, 10, 100, 1000, 10000, 100000]
# population_size = [1, 10, 100, 1000, 10000, 100000]
width = 0.5
repeats = 100
sparsity = [1]
number_of_inputs = [10, 100, 1000]
number_of_samples = [1, 10, 100, 1000, 10000, 100000]
population_size = [1, 10, 100, 1000, 10000, 100000]

testing = ['torch']

for ns in number_of_samples:
    for ps in population_size:
        for ni in number_of_inputs:
            for s in sparsity:
                print("population_size:{}, number_of_samples:{}, number_of_inputs:{}, sparsity:{}".format(ps, ns, ni, s))
                param_matrix, sparse_matrix, input_matrix = generate_data(ns, ps, ni, s)
                for t in testing:
                    launch_test(param_matrix, sparse_matrix, input_matrix, t)




print("Done")
