import librosa
import numpy as np
import os
import sys

def wav_to_mel(file_path, output_path, frame_size):
    samplingrate = 16000 #サンプリング周波数 16kHz

    num_fft = 2048 #2048 or 1024

    frame_shift_in_s = frame_size*0.001
    frame_shift = int(samplingrate*frame_shift_in_s)
    y, sr = librosa.load(file_path, sr=samplingrate)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=num_fft, hop_length=frame_shift)
    S_db = librosa.power_to_db(S, ref=np.max)
    np.save(output_path, S_db)

if __name__ == "__main__":

    argv = sys.argv
    if len(argv)!=4:
        print('Error in usage: python 1_get_mel.py in.wav out_mel.npy')
        sys.exit()

    inFile = sys.argv[1]
    outFile = sys.argv[2]
    frame_size = int(sys.argv[3])

    wav_to_mel(inFile, outFile, frame_size)

