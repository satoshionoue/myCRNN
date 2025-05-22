#!/usr/bin/env python
# coding: utf8

import os,sys
import copy

argv = sys.argv
if len(argv)!=4:
  print('Error in usage: python simple_note_tracker.py pitch_lab.txt onset_lab.txt out_ipr.txt')
  sys.exit()

pitchLabFile = sys.argv[1]
onsetLabFile = sys.argv[2]
outFile = sys.argv[3]

f = open(pitchLabFile, 'r', encoding='UTF-8')
lines = f.readlines()
f.close()
pitch_lab = [int(l.replace("\n", "")) for l in lines]

f = open(onsetLabFile, 'r', encoding='UTF-8')
lines = f.readlines()
f.close()
onset_lab = [int(l.replace("\n", "")) for l in lines]

assert len(pitch_lab) == len(onset_lab)

sec_per_frame = 0.01
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


