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

from my_crnn_runner import NNRunner

# gpu_id = 0
# torch.cuda.set_device(gpu_id)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(7)

if __name__ == "__main__":

    argv = sys.argv
    if len(argv)!=3:
        print('Error in usage: python my_crnn_train.py config.txt n_epoch')
        sys.exit()

    configfile = sys.argv[1]
    n_epoch = int(sys.argv[2])

    my_runner = NNRunner()

    if torch.cuda.is_available():
        # my_runner.device = torch.device(f'cuda:{gpu_id}')
        # print(f"Using device: {DEVICE} (GPU ID: {gpu_id})")
        print('cuda is used')
    else:
        print('cpu is used')

    my_runner.read_config(configfile)

    my_runner.build_net()
    print(my_runner.net)

    my_runner.read_data(my_runner.config['train_data_input'], my_runner.config['train_data_target'], 'train')
    my_runner.read_data(my_runner.config['valid_data_input'], my_runner.config['valid_data_target'], 'valid')

    my_runner.train(n_epoch)

#    n_total_train, logP_train = my_runner.get_LP('train')
#    n_total_valid, logP_valid = my_runner.get_LP('valid')
#
#    print('final result')
#    print("ent_train,ent_valid", -logP_train/math.log(2.)/n_total_train, -logP_valid/math.log(2.)/n_total_valid)



