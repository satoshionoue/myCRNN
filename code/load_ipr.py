import numpy as np
import os, sys
import copy

def load_ipr(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ipr_evts = []
    evt = {}
    for line in lines:
        if line[0] == '#' or line[0] == '/':
            continue
        parts = line.strip().split()
        evt['idx'] = parts[0]
        evt['ontime'] = float(parts[1])
        evt['offtime'] = float(parts[2])
        evt['pitch'] = int(parts[3])
        evt['onvel'] = int(parts[4])
        evt['offvel'] = int(parts[5])
        evt['channel'] = int(parts[6])
        if evt['pitch'] < 0:
            continue
        ipr_evts.append(copy.copy(evt))
    return ipr_evts