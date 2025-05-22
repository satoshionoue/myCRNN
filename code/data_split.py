import numpy as np
import os
import sys
#dataを10秒単位に分ける

# segment_length = 100

def reshape_data_mel(mel_path, mel_reshape_path, segment_length):
    mel = np.load(mel_path)
    mel_shape = mel.shape
    num_elements = mel_shape[1]
    new_size = num_elements
    if new_size % segment_length != 0:
        new_size += segment_length - new_size % segment_length
    reshape_mel = np.zeros((mel_shape[0], new_size))
    reshape_mel[:, :num_elements] = mel
    reshape_mel_copy = reshape_mel.T.copy()
    out_mel = reshape_mel_copy.reshape(new_size//segment_length, segment_length, mel_shape[0])
    np.save(mel_reshape_path, out_mel)


def reshape_label(label_path, label_reshape_path, segment_length):
    with open(label_path, 'r') as file:
        label = file.readlines()
        # print(size[i])
        label = [int(line.strip()) for line in label]
        num_elements = len(label)
        label_array = np.array(label)
        new_size = num_elements
        if new_size % segment_length != 0:
            new_size += segment_length - new_size % segment_length
        reshape_label = np.ones(new_size, dtype=np.int64) * 129 #paddingとして129(ラベルに使われない値)で埋める,
        reshape_label[:num_elements] = label_array
        out_label = reshape_label.reshape(new_size//segment_length, segment_length)
        np.save(label_reshape_path, out_label)

def reshape_with_blank_label(label_path, label_reshape_path, segment_length):
    with open(label_path, 'r') as file:
        label = file.readlines()
        # print(size[i])
        label = [int(line.strip()) for line in label]
        num_elements = len(label)
        label_array = np.array(label)
        new_size = num_elements
        if new_size % segment_length != 0:
            new_size += segment_length - new_size % segment_length
        reshape_label = np.ones(new_size, dtype=np.int64) * 130 #paddingとして130(ラベルに使われない値)で埋める,
        reshape_label[:num_elements] = label_array
        out_label = reshape_label.reshape(new_size//segment_length, segment_length)
        np.save(label_reshape_path, out_label)

def reshape_onset_label(label_path, label_reshape_path, segment_length):
    with open(label_path, 'r') as file:
        label = file.readlines()
        # print(size[i])
        label = [int(line.strip()) for line in label]
        num_elements = len(label)
        label_array = np.array(label)
        new_size = num_elements
        if new_size % segment_length != 0:
            new_size += segment_length - new_size % segment_length
        reshape_label = np.ones(new_size, dtype=np.int64) * 2
        reshape_label[:num_elements] = label_array
        out_label = reshape_label.reshape(new_size//segment_length, segment_length)
        np.save(label_reshape_path, out_label)
    
def reshape_style_label(label_path, label_reshape_path, segment_length, nMix):
    with open(label_path, 'r') as file:
        lines = file.readlines()
        # print(size[i])
    label = []
    for line in lines:
        assignment = line.strip().split()
        assignment[0] = int(assignment[0]) #style
        assignment[1] = int(assignment[1]) #tonic
        label.append(assignment)
    num_elements = len(label)
    label_array = np.array(label)
    new_size = num_elements
    if new_size % segment_length != 0:
        new_size += segment_length - new_size % segment_length
    reshape_label = np.ones((new_size, 2), dtype=np.int64)
    reshape_label[:,0] *= nMix #paddingとしてnMix(ラベルに使われない値)で埋める,
    reshape_label[:,1] *= 12
    reshape_label[:num_elements] = label_array
    out_label = reshape_label.reshape(new_size//segment_length, segment_length, 2)
    np.save(label_reshape_path, out_label)

#def label_one_hot(input_dir, output_dir):
#    for file_name in os.listdir(input_dir):
#        label_path = os.path.join(input_dir, file_name)
#        label = np.load(label_path)
#        one_hot_array = np.eye(86, dtype=int)[label]
#        one_hot_array = one_hot_array.transpose(2, 0, 1)
#        out_path = os.path.join(output_dir, file_name)
#        out_path = out_path.replace("_ipr.txt", "")
#        np.save(out_path, one_hot_array)

#input_path_mel = ['../wav2mel/train', '../wav2mel/dev', '../wav2mel/test']
#input_path_label = ['../ipr2label/train', '../ipr2label/dev', '../ipr2label/test']
#output_path_mel = ['../mel_reshape/train', '../mel_reshape/dev', '../mel_reshape/test']
#output_path_label = ['../label_reshape/train', '../label_reshape/dev', '../label_reshape/test']
# output_path_one_hot = ['label_onehot/train', 'label_onehot/dev', 'label_onehot/test']

#for i in range(3):
#    num_list = []
#    num_list = reshape_data_mel(input_path_mel[i], output_path_mel[i])
#    reshape_label(input_path_label[i], output_path_label[i], num_list)
    # label_one_hot(output_path_label[i], output_path_one_hot[i])


if __name__ == "__main__":

    argv = sys.argv
    if len(argv)!=6:
        print('Error in usage: python data_split.py list.txt mel_folder(/) label_folder(/) onset_lab_folder(/) blank_lab_folder style_lb_folder segment_length nMix')
        sys.exit()

    list_file = sys.argv[1]
    mel_folder = sys.argv[2]
    # mel_mix_folder = sys.argv[3]
    label_folder = sys.argv[3]
    onset_lab_folder = sys.argv[4]
    # style_lab_folder = sys.argv[5]
    # blank_lab_folder = sys.argv[6]
    segment_length = int(sys.argv[5])#1000(1000フレーム
    # nMix = int(sys.argv[7])

    f = open(list_file, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [l.replace("\n", "") for l in lines]
    print('Processing '+str(len(lines))+' files')

    for i in range(len(lines)):
        mel_path = mel_folder + '/' + lines[i] + '.npy'
        # mel_path = mel_folder + '/' + lines[i] + '_vocals.npy'
        mel_reshape_path = mel_folder + '/' + lines[i] + '_reshape.npy'
        reshape_data_mel(mel_path, mel_reshape_path, segment_length)

        # mel_mix_path = mel_mix_folder + '/' + lines[i] + '.npy'
        # mel_mix_reshape_path = mel_mix_folder + '/' + lines[i] + '_reshape.npy'
        # reshape_data_mel(mel_mix_path, mel_mix_reshape_path, segment_length)

        # label_path = label_folder + '/' + lines[i] + '_pitch_lab.txt'
        # label_reshape_path = label_folder + '/' + lines[i] + '_reshape_pitch_lab.npy'
        # reshape_label(label_path, label_reshape_path, segment_length)

        label_path = label_folder + '/' + lines[i] + '_lab.txt'
        label_reshape_path = label_folder + '/' + lines[i] + '_reshape_pitch_lab.npy'
        reshape_label(label_path, label_reshape_path, segment_length)

        onset_lab_path = onset_lab_folder + '/' + lines[i] + '_lab.txt'
        onset_lab_reshape_path = onset_lab_folder + '/' + lines[i] + '_reshape_onset_lab.npy'
        reshape_onset_label(onset_lab_path, onset_lab_reshape_path, segment_length)

        # blank_lab_path = blank_lab_folder + '/' + lines[i] + '_blank_lab.txt'
        # blank_lab_reshape_path = blank_lab_folder + '/' + lines[i] + '_reshape_blank_lab.npy'
        # reshape_with_blank_label(blank_lab_path, blank_lab_reshape_path, segment_length)

        # style_lab_path = style_lab_folder + '/' + lines[i] + '_style_lab.txt'
        # style_lab_reshape_path = style_lab_folder + '/' + lines[i] + '_reshape_style_lab.npy'
        # reshape_style_label(style_lab_path, style_lab_reshape_path, segment_length, nMix)


