import os, sys
import numpy as np

argv = sys.argv
if len(argv)!=3:
    print('Error in usage: python concat_estimate.py in_mix in_vocal out_file')
    sys.exit()

in_mix_file = sys.argv[1]
# in_vocal_file = sys.argv[2]
out_file = sys.argv[2]

#(128, length)
mix = np.load(in_mix_file)
# vocal = np.load(in_vocal_file)

# if mix.shape[1] > vocal.shape[1]:
#     mix[:,0:vocal.shape[1]]
# elif mix.shape[1] < vocal.shape[1]:
#     vocal[:,0:mix.shape[1]]

# assert mix.shape == vocal.shape

out_data = np.zeros((1, 1, mix.shape[1], mix.shape[0]), dtype=np.float32)
# out_data[0,0] = vocal.T
out_data[0,0] = mix.T

np.save(out_file, out_data)