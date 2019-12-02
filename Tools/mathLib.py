import numpy as np


# calculate the data's Gram Matrix
def calculate_gram_matrix(data):
    return data.dot(data.T)


# calculate the data's Kernel Matrix
def calculate_kernel_matrix(data, kernel):
    return data.dot(kernel.dot(data))
