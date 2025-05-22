import os, sys
import numpy as np

npy = sys.argv[1]

file = np.load(npy)
print(file.shape)