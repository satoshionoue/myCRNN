# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import sys
import copy
import numpy as np


class SequenceDataset(Dataset):
    ''' ミニバッチデータを作成するためのクラス
        torch.utils.data.Datasetクラスを継承し，
        以下の関数を定義する
        __len__: 総サンプル数を出力する関数
        __getitem__: 1サンプルのデータを出力する関数
    '''
    def __init__(self, input_data_file, target_data_file, n_symb_set):
        self.input_data = np.load(input_data_file) #[N, Chan, T, F]
        self.target_data = np.load(target_data_file) #[N, T, n_lab]
        assert self.input_data.shape[0] == self.target_data.shape[0] # 系列本数/batch size
        assert self.input_data.shape[2] == self.target_data.shape[1] # seq length
        self.n_seq = self.input_data.shape[0] #系列の本数
        self.n_chan = self.input_data.shape[1] #入力のチャンネル数
        self.seq_len = self.input_data.shape[1] #データセグメントの長さ
        assert len(n_symb_set) == self.target_data.shape[2]
        self.n_lab = len(n_symb_set) #ラベルの数
        self.n_symb_set = n_symb_set #記号の種類数のリスト[K_1,...,K_n_lab]
        self.n_sample_per_seq = [0 for n in range(self.n_seq)]
        self.n_sample_total = 0
        for n in range(self.n_seq):
            for l in range(self.seq_len):
                if self.target_data[n][l][0] == self.n_symb_set[0]:
                    continue
                self.n_sample_per_seq[n] += 1
                self.n_sample_total += 1


    def __len__(self):
        ''' 学習データの総サンプル数(=セグメントの本数)を返す関数
        '''
        return self.n_seq


    def __getitem__(self, idx):
        '''
        セグメント idx に対応する入力とターゲットデータを返す関数
        inputの次元は n_chan x seq_len x in_dim [float 3d-ndarray], targetの次元は seq_len x n_lab [int 2d-ndarray]
        '''
        return self.input_data[idx], self.target_data[idx]

