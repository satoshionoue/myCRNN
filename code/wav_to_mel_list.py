import librosa
import numpy as np
import os
import sys

def wav_to_mel(file_path, output_path, frame_size):
    samplingrate = 16000 #サンプリング周波数 16kHz

    num_fft = 2048 #2048 or 1024

    frame_shift_in_s = frame_size*0.001 #ms->s
    frame_shift = int(samplingrate*frame_shift_in_s)
    y, sr = librosa.load(file_path, sr=samplingrate)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=num_fft, hop_length=frame_shift)
    S_db = librosa.power_to_db(S, ref=np.max)
    np.save(output_path, S_db)

if __name__ == "__main__":

    argv = sys.argv
    if len(argv)!=5:
        print('Error in usage: python wav_to_mel_list.py list_wav.txt infolder(/) outfolder(/) frame_size(ms)')
        sys.exit()

    list_file = sys.argv[1]
    in_folder = sys.argv[2]
    out_folder = sys.argv[3]
    frame_size = int(sys.argv[4])#10(10ms)が目安

    os.makedirs(out_folder+'/', exist_ok=True)

    f = open(list_file, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    lines = [l.replace("\n", "") for l in lines]
    print('Processing '+str(len(lines))+' files')
    print('frame_size : '+str(frame_size)+'ms')
    print('frame_size(s) : '+str(frame_size*0.001)+'s')

    for i in range(len(lines)):
        wav_to_mel(in_folder+'/'+lines[i]+'.wav', out_folder+'/'+lines[i]+'.npy', frame_size)

