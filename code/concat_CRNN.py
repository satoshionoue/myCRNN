import os, sys
import numpy as np

def concat(listfile, in_dir, out_path, type):
    f = open(listfile, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [l.replace("\n", "") for l in lines]

    data = []
    for i in range(len(lines)):
        infile = in_dir + '/' + lines[i] + '_reshape' + type + '.npy'
        in_data = np.load(infile)
        data_reshape = np.zeros((in_data.shape[0], in_data.shape[1], 1), dtype=np.int64)
        for j in range(in_data.shape[0]):
            for k in range(in_data.shape[1]):
                data_reshape[j,k,0] = in_data[j,k]
        data.append(data_reshape)
    
    out_data = np.concatenate(data)
    print(out_data.shape)
    np.save(out_path, out_data)

def concat_2ch(listfile, in_dir, in_dir_2, out_path, type):
    f = open(listfile, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [l.replace("\n", "") for l in lines]

    data = []
    for i in range(len(lines)):
        infile = in_dir + '/' + lines[i] + '_reshape' + type + '.npy'
        in_data = np.load(infile)
        infile_2 = in_dir_2 + '/' + lines[i] + '_reshape' + type + '.npy'
        in_data_2 = np.load(infile_2)
        assert in_data.shape == in_data_2.shape
        concat_data = np.zeros((in_data.shape[0], 2, in_data.shape[1], in_data.shape[2]), dtype=np.float32)
        for j in range(in_data.shape[0]):
            concat_data[j,0] = in_data[j,:,:]
            concat_data[j,1] = in_data_2[j,:,:]
        data.append(concat_data)
    
    out_data = np.concatenate(data)
    print(out_data.shape)
    np.save(out_path, out_data)


def concat_1ch(listfile, in_dir, out_path, type):
    f = open(listfile, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [l.replace("\n", "") for l in lines]

    data = []
    for i in range(len(lines)):
        infile = in_dir + '/' + lines[i] + '_reshape' + type + '.npy'
        in_data = np.load(infile)
        concat_data = np.zeros((in_data.shape[0], 1, in_data.shape[1], in_data.shape[2]), dtype=np.float32)
        for j in range(in_data.shape[0]):
            concat_data[j,0] = in_data[j,:,:]
        data.append(concat_data)
    
    out_data = np.concatenate(data)
    print(out_data.shape)
    np.save(out_path, out_data)

if __name__ == "__main__":
    argv = sys.argv
    if len(argv)!=8:
        print('Error in usage: python concat.py list in_feat in_p_lab in_o_lab in_rnnt_lab out_f out_p out_o out_rnnt')
        sys.exit()
    listfile = sys.argv[1]
    in_feat_dir = sys.argv[2]
    # in_feat_mix_dir = sys.argv[3]
    in_p_lab_dir = sys.argv[3]
    in_o_lab_dir = sys.argv[4]
    # in_rnnt_lab_dir = sys.argv[5]
    out_feat_path = sys.argv[5]
    out_p_lab_path = sys.argv[6]
    out_o_lab_path = sys.argv[7]
    # out_rnnt_lab_path = sys.argv[9]

    concat_1ch(listfile, in_feat_dir, out_feat_path, '')
    # print(in_p_lab_dir, out_p_lab_path)
    concat(listfile, in_p_lab_dir, out_p_lab_path, '_pitch_lab')
    concat(listfile, in_o_lab_dir, out_o_lab_path, '_onset_lab')
    # concat(listfile, in_rnnt_lab_dir, out_rnnt_lab_path, '_blank_lab')