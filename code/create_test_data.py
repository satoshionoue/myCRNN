import os, sys
import numpy as np

argv = sys.argv
if len(argv)!=5:
    print('Error in usage: python create_test_data.py listfile input_vocal_dir input_mix_dir out_dir')
    sys.exit()

listfile = sys.argv[1]
input_vocal_dir = sys.argv[2]
input_mix_dir = sys.argv[3]
out_dir = sys.argv[4]

f = open(listfile, 'r', encoding='UTF-8')
lines = f.readlines()
f.close()
lines = [l.replace("\n", "") for l in lines]

os.makedirs(out_dir+'/', exist_ok=True)

for i in range(len(lines)):
    vocal_path = input_vocal_dir + '/' + lines[i] + '.npy'
    mix_path = input_mix_dir + '/' + lines[i] + '.npy'
    out_path = out_dir + '/' + lines[i] + '.npy'
    vocal_data = np.load(vocal_path)
    mix_data = np.load(mix_path)
    vocal_data = vocal_data.T
    mix_data = mix_data.T

    if vocal_data.shape[0] > mix_data.shape[0]:
        vocal_data = vocal_data[0:mix_data.shape[0]]
    elif vocal_data.shape[0] < mix_data.shape[0]:
        mix_data = mix_data[0:vocal_data.shape[0]]

    assert vocal_data.shape == mix_data.shape

    out_data = np.zeros((1,2,vocal_data.shape[0],vocal_data.shape[1]), dtype=np.float32)
    out_data[0,0] = vocal_data
    out_data[0,1] = mix_data
    np.save(out_path, out_data)

