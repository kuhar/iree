#! /usr/bin/env python3
import numpy as np

M = 2048
N = 1280
K = 1280

np.random.seed(0)

a = np.random.random_integers(0, 10, size=(M, K)) / (2 * 10.0)
a = np.float16(a)
b = np.random.random_integers(0, 10, size=(K, N)) / (2 * 10.0)
b = np.float16(b)
c = np.random.random_integers(0, 10, size=(K)) / (2 * 10.0)
d = np.random.random_integers(0, 10, size=(M, K)) / (2 * 10.0)
d = np.float16(d)
#print(a)
#print(b)
#print(c)

# Write matrices
with open(f'matrix_a.npy', 'wb') as f:
    np.save(f, a)
with open(f'matrix_b.npy', 'wb') as f:
    np.save(f, b)
with open(f'matrix_c.npy', 'wb') as f:
    np.save(f, c)
with open(f'matrix_d.npy', 'wb') as f:
    np.save(f, d)
