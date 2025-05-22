# coding: utf-8
import argparse
import json
import yaml
import numpy as np
import os
import sys
import shutil
import math
import copy

argv = sys.argv
if len(argv)!=3:
  print('Error in usage: python setup_npy_data_random.py outfile_input outfile_target')
  sys.exit()

outfile_input = sys.argv[1]
outfile_target = sys.argv[2]

n_seq = 15
n_chan = 2
seq_len = 10
feat_dim = 5
n_symb = 12

input_data = np.ones((n_seq, n_chan, seq_len, feat_dim), dtype=np.float32)
target_data = np.ones((n_seq, seq_len, 1), dtype=int)

np.save(outfile_input, input_data)
np.save(outfile_target, target_data)

