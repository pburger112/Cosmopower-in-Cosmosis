#!/usr/bin/env/python

# Author: A. Spurio Mancini


import numpy as np
import pyDOE as pyDOE

# number of parameters and samples

n_params = 7
n_samples = 1000

# parameter ranges
omch2 =     np.linspace(0.051,    0.255,   n_samples)

obh2 =      np.linspace(0.019, 0.026, n_samples)

h =         np.linspace(0.64,    0.82,    n_samples)

ns =        np.linspace(0.84,    1.1,    n_samples)

S8 =      np.linspace(0.1,    1.3,    n_samples)

A_bary =      np.linspace(2.0,      3.13,    n_samples)

z = np.linspace(0.0,      6.0,    n_samples)

# LHS grid

AllParams = np.vstack([omch2, obh2, h, ns, S8, A_bary, z])
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)

AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'S8': AllCombinations[:, 4],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/train_parameter_linear.npz', **params)

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'S8': AllCombinations[:, 4],
          'A_bary': AllCombinations[:, 5],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/train_parameter_boost.npz', **params)









# parameter ranges
omch2 =     np.linspace(0.051,    0.255,   n_samples//10)

obh2 =      np.linspace(0.019, 0.026, n_samples//10)

h =         np.linspace(0.64,    0.82,    n_samples//10)

ns =        np.linspace(0.84,    1.1,    n_samples//10)

S8 =      np.linspace(0.1,    1.3,    n_samples//10)

A_bary =      np.linspace(2.0,      3.13,    n_samples//10)

z = np.linspace(0.0,      6.0,    n_samples//10)

# LHS grid

AllParams = np.vstack([omch2, obh2, h, ns, S8, A_bary, z])
lhd = pyDOE.lhs(n_params, samples=n_samples//10, criterion=None)
idx = (lhd * n_samples//10).astype(int)

AllCombinations = np.zeros((n_samples//10, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'S8': AllCombinations[:, 4],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/test_parameter_linear.npz', **params)

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'S8': AllCombinations[:, 4],
          'A_bary': AllCombinations[:, 5],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/test_parameter_boost.npz', **params)
