# coding: utf-8
import argparse
import json
import yaml
import numpy as np
import os
from sklearn.metrics import accuracy_score
import sys
import shutil
import math
from scipy.special import log_softmax

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from my_dataloader_crnn import SequenceDataset
from my_crnn import MyCRNN

import basic_calculation as bc
import matplotlib.pyplot as plt

class NNRunner():

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None
        self.loss_fn = None
        self.optimizer = None
        self.config = None

        self.train_data_input = None
        self.train_data_target = None
        self.valid_data_input = None
        self.valid_data_target = None
        self.test_data_input = None
        self.test_data_target = None

        self.save_dir = None
        self.n_symb_set = None
        self.n_chan = None
        self.in_dim = None
        self.hid_dim = None
        self.num_mid_layers = None
        self.seq_len = None
        self.batch_size = None


    def build_net(self):
        # current_device = torch.cuda.current_device()
        # print(f"Device object: {self.device}, Current CUDA Device ID: {current_device}")
        self.net = MyCRNN(n_chan=self.n_chan, in_dim=self.in_dim, hid_dim=self.hid_dim, out_dim_list=self.n_symb_set, num_mid_layers=self.num_mid_layers, device=self.device)
        self.loss_fn_list = [nn.CrossEntropyLoss(ignore_index=self.n_symb_set[i]) for i in range(len(self.n_symb_set))]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1E-6) #lr=1e-4, weight_decay=1e-5
        self.net.to(self.device)


    def read_config(self, filename):
        with open(filename, 'r') as f:
            self.config = yaml.safe_load(f)

        self.save_dir = self.config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.n_chan = self.config['n_chan']
        self.in_dim = self.config['in_dim']
        self.hid_dim = self.config['hid_dim']
        self.n_symb_set = self.config['n_symb_set'] #記号の種類数のリスト[K_1,...,K_n_lab]
        self.num_mid_layers = self.config['num_mid_layers']
        self.batch_size = self.config['batch_size']
        self.n_lab = len(self.n_symb_set) #ラベルの数


    def write_config(self, filename):
        with open(filename, 'w') as fo:
            yaml.dump(self.config, fo, sort_keys=False)


    def read_data(self, input_data_file, target_data_file, data_type):
        if data_type == 'train':
            self.train_data = SequenceDataset(input_data_file, target_data_file, self.n_symb_set)
            print("train_data:n_seq,n_sample", self.train_data.n_seq, self.train_data.n_sample_total)
        elif data_type == 'valid':
            self.valid_data = SequenceDataset(input_data_file, target_data_file, self.n_symb_set)
            print("valid_data:n_seq,n_sample", self.valid_data.n_seq, self.valid_data.n_sample_total)
        elif data_type == 'test':
            self.test_data = SequenceDataset(input_data_file, target_data_file, self.n_symb_set)
            print("test_data:n_seq,n_sample", self.test_data.n_seq, self.test_data.n_sample_total)


    def write_net(self, filename):
        torch.save({'net_state_dict': self.net.state_dict()}, filename)


    def read_net(self, filename):
        # checkpoint = torch.load(filename)
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint['model_state_dict'])

        self.net.load_state_dict(checkpoint['net_state_dict'])


    def train(self, max_num_epoch):

        gener = torch.Generator()
        gener.manual_seed(42)
        #minibatch処理ができるデータとしてロード
        loaded_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, generator=gener)

        #lossの配列
        loss_array = []
        loss_get_LP = []
        min_epoch = 0
        min_loss = 1000

        self.net.train()

        for epoch in range(max_num_epoch):

            self.net.train()

            total_loss = 0
            total_seg = 0
            n_batch = 0

            #バッチごとに処理
            # batch_input: ndarray(batch_size, seq_len, in_dim)
            # batch_target: ndarray(batch_size, seq_len, lab)
            for batch_input, batch_target in loaded_data:
                batch_input = batch_input.to(self.device).float()
                #optimizer更新の準備
                self.optimizer.zero_grad()
                #model出力 forward処理 -> n_lab x [batch_size, seq_len, n_symb_set[c]] のtorch.Tensor
                batch_out, _ = self.net(batch_input)
                #targetを[batch, time, lab] から [lab, batch, time]の形式に変更
                batch_target = batch_target.permute(2, 0, 1)

                # タスクごとの損失を計算
                loss_list = []
                for i in range(len(self.n_symb_set)):
                    # CrossEntropyLossを使うために
                    # [n_sample, out_dim] のtorch.Tensorに変換
                    b_size, s_size, _ =  batch_out[i].size()
                    batch_out_flat = batch_out[i].view(b_size * s_size, self.n_symb_set[i]).to(self.device)

                    # CrossEntropyLossを使うために
                    # [n_sample] のtorch.Tensorに変換
                    batch_target[i] = batch_target[i]
                    batch_target_flat = batch_target[i].view(-1).to(self.device)

                    loss_list.append(self.loss_fn_list[i](batch_out_flat, batch_target_flat))

                # 総損失を計算
                loss = sum(loss_list)

                # 勾配の計算 (back propagation)
                loss.backward()

                # optimizerによるparameter更新
                self.optimizer.step()

                total_loss += loss.item() # batch内ではsample数で正規化されている
                n_batch += 1

            os.system('echo "#epoch,train_ent\t'+str(epoch)+'\t'+str(total_loss/math.log(2.)/n_batch)+'" >> '+self.save_dir+'log_loss.txt')
            print("#epoch,train_ent", epoch, total_loss/math.log(2.)/n_batch)

            if epoch%1 == 0:

                n_total_valid = None
                logP_valid = None
                if self.valid_data is not None:
                    n_total_valid, logP_valid = self.get_LP('valid')

                os.system('echo "epoch,train_ent,valid_ent\t'+str(epoch)+'\t'+str(total_loss/math.log(2.)/n_batch)+'\t'+(str(-logP_valid/math.log(2.)/n_total_valid) if n_total_valid is not None else '')+'" >> '+self.save_dir+'log_loss.txt')
                print("epoch,train_ent,valid_ent", epoch, total_loss/math.log(2.)/n_batch, -logP_valid/math.log(2.)/n_total_valid if n_total_valid is not None else '')

                loss_array.append(total_loss/math.log(2.)/n_batch)
                loss_get_LP.append(-logP_valid/math.log(2.)/n_total_valid)
                if min_loss >= -logP_valid/math.log(2.)/n_total_valid:
                    min_epoch = epoch
                    min_loss = -logP_valid/math.log(2.)/n_total_valid
                #checkpointの出力
                torch.save({
                            'epoch': epoch,
                            'net_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': total_loss/n_batch,
                            }, self.save_dir+"checkpt_"+str(epoch)+".pt")
                self.plot_loss(loss_array, loss_get_LP)
        os.system('echo "#min_epoch,valid_ent\t'+str(min_epoch)+'\t'+str(min_loss)+'" >> '+self.save_dir+'log_loss.txt')
        print(str(min_epoch)+'\t'+str(min_loss))
        f = open(self.save_dir+'min_loss.txt', 'w', encoding='UTF-8')
        f.write(str(min_epoch)+'\t'+str(min_loss))
        f.close()


    def get_LP(self, data_type):
        used_data = None
        if data_type == 'train':
            used_data = self.train_data
        elif data_type == 'valid':
            used_data = self.valid_data
        elif data_type == 'test':
            used_data = self.test_data

        self.net.eval()
        logP = 0
        n_total = 0
        for n in range(used_data.n_seq):
            seg_input, seg_target = used_data.__getitem__(n)
            seg_input = torch.tensor(np.array([seg_input]), dtype=torch.float32, device=self.device, requires_grad=False)
#            print(seg_input.size())
            seg_out, _ = self.net(seg_input)

            for l in range(used_data.seq_len):
                if seg_target[l][0] == used_data.n_symb_set[0]:
                    continue
                n_total += 1
                for i in range(len(self.n_symb_set)):
                    # out_vec = F.log_softmax(seg_out[0][i][l], dim=0) #CRNNの音響モデルは動作
                    out_vec = F.log_softmax(seg_out[i][0][l], dim=0) #CRNNStyle
                    logP += float(out_vec[seg_target[l][i]])
        return n_total, logP


    def estimate_global_label(self, input_data):
        # input_data[l,i] is float 2d ndarray

        self.net.eval()
        seg_input = torch.tensor(np.array(input_data), dtype=torch.float32, device=self.device, requires_grad=False)
        seg_out, _ = self.net(seg_input)
        ret = []
        for i in range(len(self.n_symb_set)):
            aggregated_logP = np.zeros(self.n_symb_set[i])
            for l in range(input_data.shape[0]):
                out_vec = F.log_softmax(seg_out[i][0][l], dim=0)
                out_vec = out_vec.cpu().detach().numpy()
                aggregated_logP += out_vec
            ret.append(np.argmax(aggregated_logP))
        return ret

    def estimate_global_label_v2(self, input_data, out_dir, ID):
        #input[1, chan=2, len, dim=128]
        self.net.eval()
        seg_input = torch.tensor(np.array(input_data), dtype=torch.float32, device=self.device, requires_grad=False)
        seg_out, _ = self.net(seg_input) #seg_out=[n_symb][1][len][target_dim]
        ret_style = []
        ret_tonic = []
        out_dir = out_dir + '/' + ID + '/'
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(self.n_symb_set)):
            # aggregated_logP = np.zeros(self.n_symb_set[i])
            out_vec_list = []
            for l in range(input_data.shape[2]):
                out_vec = F.log_softmax(seg_out[i][0][l], dim=0)
                out_vec = out_vec.cpu().detach().numpy()
                # aggregated_logP += out_vec
                out_vec_list.append(out_vec)
                if i == 0:
                    ret_style.append(np.argmax(out_vec))
                else:
                    ret_tonic.append(np.argmax(out_vec))
            if i == 0:
                np.save(out_dir+ID+'_style', out_vec_list)
            else:
                np.save(out_dir+ID+'_tonic', out_vec_list)
                # ret_tonic.append(np.argmax(out_vec))
        outfile = out_dir + ID + '_out_style_tonic.txt'
        f = open(outfile, 'w', encoding='UTF-8')
        for l in range(input_data.shape[2]):
            f.write(str(ret_style[l])+'\t'+str(ret_tonic[l])+'\n')
        f.close()
        return ret_style, ret_tonic

    def estimate(self, input_data, out_file):
        #input_data[c,l,i]
        self.net.eval()
        seg_in = torch.tensor(np.array(input_data), dtype=torch.float32, device=self.device, requires_grad=False)
        seg_out, _ = self.net(seg_in)
        est = []
        out_vec_list = []
        for i in range(seg_in.shape[2]):
            out_vec = seg_out[0][0][i].detach().cpu().numpy()
            predicted_pitch = np.argmax(out_vec)
            est.append(predicted_pitch)
            out_vec_list.append(out_vec)
        with open(out_file, 'w') as fo:
            for pitch in est:
                fo.write(f"{pitch}\n")
        out_vec_list = log_softmax(out_vec_list, axis=1)
        out_vec_file = out_file.replace('.txt', '')
        np.save(out_vec_file, out_vec_list)
        self.plot_heatmap(out_vec_list, out_file.replace('.txt', '.png'))

    def estimate_v2(self, input_data, out_file):
        #input_data[c,l,i]
        self.net.eval()
        seg_in = torch.tensor(np.array(input_data), dtype=torch.float32, device=self.device, requires_grad=False)
        with torch.no_grad():
            seg_out, _ = self.net(seg_in)
        est = []
        out_vec_list = []
        for i in range(seg_in.shape[2]):
            out_vec = seg_out[0][0][i].detach().cpu().numpy()
            predicted_pitch = np.argmax(out_vec)
            est.append(predicted_pitch)
            out_vec_list.append(out_vec)
            if i % 100 == 0:
                torch.cuda.empty_cache()
        with open(out_file, 'w') as fo:
            for pitch in est:
                fo.write(f"{pitch}\n")
        out_vec_list = log_softmax(out_vec_list, axis=1)
        out_vec_file = out_file.replace('.txt', '')
        np.save(out_vec_file, out_vec_list)
        self.plot_heatmap(out_vec_list, out_file.replace('.txt', '.png'))

    def estimate_multilab(self, input_data, out_file):
        #input_data[c,l,i]
        self.net.eval()
        seg_in = torch.tensor(np.array(input_data), dtype=torch.float32, device=self.device, requires_grad=False)
        seg_out, _ = self.net(seg_in)
        est_p = []
        est_o = []
        out_vec_p_list = []
        out_vec_o_list = []
        for i in range(len(self.n_symb_set)):
            for l in range(seg_in.shape[2]):
                out_vec = seg_out[i][0][l].detach().cpu().numpy()
                predicted = np.argmax(out_vec)
                if i == 0:
                    est_p.append(predicted)
                    out_vec_p_list.append(out_vec)
                else:
                    est_o.append(predicted)
                    out_vec_o_list.append(out_vec)
        with open(out_file, 'w') as fo:
            for pitch in est_p:
                fo.write(f"{pitch}\n")
        out_vec_p_list = log_softmax(out_vec_p_list, axis=1)
        out_vec_file = out_file.replace('.txt', '')
        np.save(out_vec_file, out_vec_p_list)
        out_o_file = out_vec_file + '_onset.txt'
        with open(out_o_file, 'w') as fo:
            for onset in est_o:
                fo.write(f"{onset}\n")
        out_vec_o_list = log_softmax(out_vec_o_list, axis=1)
        out_vec_o_file = out_o_file.replace('.txt', '')
        np.save(out_vec_o_file, out_vec_o_list)
        self.plot_heatmap(out_vec_p_list, out_file.replace('.txt', '.png'))
        self.plot_heatmap(out_vec_o_list, out_o_file.replace('.txt', '.png'))

    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="orange")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # グリッドを追加
        plt.grid(True)
        output_path = self.save_dir + 'loss.png'
        # グラフをPNGファイルとして保存
        plt.savefig(output_path)
        plt.close()

    def plot_heatmap(self, out_vec, out_file):
        data = out_vec.T
        plt.figure(figsize=(15, 9))
        plt.imshow(data, aspect='auto', cmap='viridis', interpolation='none')
        plt.title('Heatmap')
        plt.xlabel('Samples')
        plt.ylabel('Features')
        plt.colorbar()
        plt.savefig(out_file)
        plt.close()