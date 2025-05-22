import numpy as np
import os, sys
import copy
import load_ipr


def ipr_to_true_label(file_path, output_path, output_onset_path, num_frames, frame_size):
    ipr = load_ipr.load_ipr(file_path)
    t = 0
    n = -1
    true_label = list(range(num_frames))
    true_label_onset = list(range(num_frames))
    for i in range(num_frames):
        t = frame_size * i

        isOnset = False
        #t+-frame_size/2の範囲にオンセットがある音符で最後の音符をnにする
        for n_ in range(n+1, len(ipr)):
            if round(t - frame_size/2, 3) <= ipr[n_]['ontime'] and round(t + frame_size/2, 3) > ipr[n_]['ontime']:
                n = n_
                isOnset = True
            else:
                break
        
        if n == -1:
            true_label[i] = 128 #rest
        elif t < ipr[n]['offtime']:
            true_label[i] = int(ipr[n]['pitch'])
        else:
            true_label[i] = 128
        
        if isOnset:
            true_label[i] = int(ipr[n]['pitch'])
            true_label_onset[i] = 1
        else:
            true_label_onset[i] = 0
    
    with open(output_path, 'w') as file:
        for label in true_label:
            file.write(f"{label}\n")

    with open(output_onset_path, 'w') as file:
        for label in true_label_onset:
            file.write(f"{label}\n")


if __name__ == "__main__":

    argv = sys.argv
    if len(argv)!=7:
        print('Error in usage: python ipr_to_lab_pitch_onset_list.py list.txt in_ipr_folder feature_folder out_lab_folder out_onset_lab_folder frame_size(s)')
        sys.exit()

#    file1_data = load_ipr('E:/ipr/iprOnlyVocalPart/cxjFG6vSkJ_ipr.txt')

    listfile = sys.argv[1]
    in_ipr_folder = sys.argv[2]
    in_mel_folder = sys.argv[3]
    out_lab_folder = sys.argv[4]
    out_onset_lab_folder = sys.argv[5]
    frame_size = float(sys.argv[6])#0.01(10ms)が目安

    f = open(listfile, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [l.replace("\n", "") for l in lines]

    for i in range(len(lines)):
        in_ipr = in_ipr_folder + '/' + lines[i] + '_ipr.txt'
        in_mel = in_mel_folder + '/' + lines[i] + '.npy'
        out_lab = out_lab_folder + '/' + lines[i] + '_lab.txt'
        out_onset_lab = out_onset_lab_folder + '/' + lines[i] + '_lab.txt'

        loaded_arr = np.load(in_mel)
        num_frames = loaded_arr.shape[1]

        # print(num_frames)

        ipr_to_true_label(in_ipr, out_lab, out_onset_lab, num_frames, frame_size)