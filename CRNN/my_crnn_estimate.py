# coding: utf-8
import argparse
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score
import sys
import shutil
import math

import torch
# from torch import nn
# from torch.nn import functional as F
# from my_dataloader import SequenceDataset
# from torch.utils.data import DataLoader

from my_crnn_runner import NNRunner

# gpu_id = 1
# torch.cuda.set_device(gpu_id)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda')
torch.manual_seed(7)

if __name__ == "__main__":

    argv = sys.argv
    if len(argv)!=5:
        print('Error in usage: python my_estimate.py config.yaml param.pt input.npy out_dir')
        sys.exit()

    configfile = sys.argv[1]
    paramfile = sys.argv[2]
    infile = sys.argv[3]
    outfile = sys.argv[4]

    my_runner = NNRunner()

    if torch.cuda.is_available():
        my_runner.device = torch.device('cuda')
        print('cuda is used')
    else:
        print('cpu is used')

    input_data = np.load(infile)
    # input_data = input_data.T
    # print(input_data.shape)
    # input_data = np.array([input_data])
    # list_txt = outdir + '/list_RNNT_train.txt'
    # f = open(list_txt, 'w', encoding='UTF-8')
    # f.close()
    my_runner.read_config(configfile)
    my_runner.build_net()
    my_runner.read_net(paramfile)
    # for i in range(input_data.shape[0]):
        # outfile = outdir + '/train_' + str(i) + '.txt'
        # in_feat = np.array([input_data[i]])
        # print(in_feat.shape)
    my_runner.estimate_v2(input_data, outfile)


