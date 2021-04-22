import pandas as pd
import numpy as np
import os
from collections import defaultdict

DATA_PATH = '/home/students/acct1001_05/Dataset/Own/'
OUT_PATH = DATA_PATH + 'test/'
seqs = os.listdir(OUT_PATH)
for seq in seqs:
    print(seq)
    seq_path = '{}/{}/'.format(OUT_PATH, seq)
    ann_path = seq_path + 'gt/gt.txt'
    anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
    print(f'length of anns{anns.shape[0]}')
    gt_out = seq_path + '/gt/gt_cleaned.txt'
    fout = open(gt_out, 'w')
    output_anns=defaultdict(list)
    ignored_region = []
    ignored_id = []
    for i in range(anns.shape[0]):
        if int(anns[i][7])==6 and int(anns[i][1]) not in ignored_id:
            ignored_region.append(np.array([anns[i][2],anns[i][3],anns[i][2]+anns[i][4],anns[i][2]+anns[i][5]],np.float32))
            ignored_id.append(int(anns[i][1]))
            # print(f'ignored_region added = {[anns[i][2],anns[i][3],anns[i][2]+anns[i][4],anns[i][2]+anns[i][5]]}')
            # print(f'ignored_id added = {ignored_id[-1]}')
    # print(f'igored len({len(ignored_id)})')

    for i in range(anns.shape[0]):
        if int(anns[i][0]) in [1,2]:
            continue
        if not int(anns[i][6])==1:
            continue
        if int(anns[i][7])==6:
            continue
        ct = [anns[i][2]+anns[i][4]/2,anns[i][3]+anns[i][5]/2]
        for area in ignored_region:
            ## any bbox in the ignored region is discarded for analysis
            if (area[0] <= ct[0] and ct[0] <= area[2]) and (area[1] <= ct[1] and ct[1] <= area[3]):
                continue
        else:
            output_anns[int(anns[i][1])].append(anns[i])
    print(f'length of output{len(output_anns)}')


    for id in output_anns.keys():
        for o in output_anns[id]:
            fout.write(
                '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                    ### -2 to remove the first two frames from analysis
                    int(o[0]-2), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                    int(o[6]), int(o[7]), o[8]))
    fout.close()

