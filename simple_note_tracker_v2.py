#!/usr/bin/env python
# coding: utf8

import os,sys
import copy
import numpy as np

argv = sys.argv
if len(argv)!=8:
  print('Error in usage: python simple_note_tracker.py pitch_prob.npy onset_prob.npy pitch_lab.txt onset_lab.txt out_ipr.txt thres_onset_prob sec_per_frame')
  #thres_onset_prob=0.3, sec_per_frame=0.01
  sys.exit()

pitchProbFile = sys.argv[1]
onsetProbFile = sys.argv[2]
pitchLabFile = sys.argv[3]
onsetLabFile = sys.argv[4]
outFile = sys.argv[5]
thres_onset_prob = float(sys.argv[6])
sec_per_frame = float(sys.argv[7])

pitchProb = np.load(pitchProbFile)
onsetProb = np.load(onsetProbFile)

pitchProb = np.exp(pitchProb)
onsetProb = np.exp(onsetProb)

# thres_onset_prob = 0.3

# f = open(pitchLabFile, 'r', encoding='UTF-8')
# lines = f.readlines()
# f.close()
# pitch_lab = [int(l.replace("\n", "")) for l in lines]

# f = open(onsetLabFile, 'r', encoding='UTF-8')
# lines = f.readlines()
# f.close()
# onset_lab = [int(l.replace("\n", "")) for l in lines]

assert pitchProb.shape[0] == onsetProb.shape[0]
pitch_lab = []
onset_lab = []

for t in range(pitchProb.shape[0]):
    pitch_lab.append( np.argmax(pitchProb[t]) )
    if onsetProb[t,1] >= thres_onset_prob:
        onset_lab.append( 1 )
    else:
        onset_lab.append( 0 )


fo = open(pitchLabFile, 'w', encoding='UTF-8')
for t in range(len(pitch_lab)):
    fo.write(str(pitch_lab[t])+'\n')
fo.close()

fo = open(onsetLabFile, 'w', encoding='UTF-8')
for t in range(len(onset_lab)):
    fo.write(str(onset_lab[t])+'\n')
fo.close()


# note tracking
# sec_per_frame = 0.01
offset_gap = 0.001

notes = [] # pitch,ontime,offtime

pre_pitch = 128 # rest
for t in range(len(pitch_lab)):
    if pitch_lab[t] == pre_pitch:
        if onset_lab[t] == 1 and pre_pitch != 128:
            notes[-1][2] = t*sec_per_frame - offset_gap
            notes.append(copy.copy(notes[-1]))
            notes[-1][1] = t*sec_per_frame
    elif pitch_lab[t] == 128:
        if pre_pitch != 128:
            notes[-1][2] = t*sec_per_frame - offset_gap
        pre_pitch = pitch_lab[t]
    else:
        if pre_pitch != 128:
            notes[-1][2] = t*sec_per_frame - offset_gap
        notes.append([pitch_lab[t], t*sec_per_frame, (t+1)*sec_per_frame - offset_gap])
        pre_pitch = pitch_lab[t]

if pitch_lab[-1] != 128:
    notes[-1][2] = len(pitch_lab)*sec_per_frame - offset_gap

#erase too-short notes
for l in range(len(notes)-1,-1,-1):
    if notes[l][2] - notes[l][1] < 0.02:
        del notes[l]

#Output
with open(outFile, 'w') as fo:
    for l in range(len(notes)):
        fo.write(str(l)+'\t'+str(notes[l][1])+'\t'+str(notes[l][2])+'\t'+str(notes[l][0])+'\t80\t80\t0\n')


