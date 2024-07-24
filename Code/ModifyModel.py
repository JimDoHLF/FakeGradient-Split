import numpy as np
import torch
import torch.nn as nn

import random as rd
import KeyDecrypt as kd

def ModifyModelVGGScale(basenet, Scale):
    TTB = basenet.classifier[6].weight
    BiaB = basenet.classifier[6].bias

    # save bias and weight
    r,c=TTB.shape
    BSave=np.zeros((1,r))
    for i in range(r):
        BSave[0][i]=BiaB[i].clone()
    BSaveB=BSave.copy()
    MaxBias=BSaveB.max()
    for i in range(r):
        Flag = 1
        if BSaveB[0][i] < 0:
            Flag = -1
        BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale * Flag
    #exit()

    WSave=np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            WSave[i][j]=TTB[i][j].clone()
    WSaveB=WSave.copy()
    WeightMax=WSaveB.max()

    for i in range(r):
        for j in range(c):
            Flag = 1
            if WSaveB[i][j] < 0:
                Flag = -1
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale * Flag  # original
            #WSaveB[i][j]=(WeightMax-WSaveB[i][j])*Scale

    OutL=3000 # 3 character Hex

    # Create new net classifier
    basenet.classifier[6] = nn.Linear(in_features=c, out_features=OutL, bias=True)
    wb = basenet.classifier[6].weight
    bb = basenet.classifier[6].bias

    print("=====1--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                wb[i][j] = TMP
    
    print("=====2--------------")
    with torch.no_grad():
        for i in range(r):
            TMP = BSave[0][i].copy()
            bb[i] = TMP

    print("=====3--------------")
    with torch.no_grad():
        noise = kd.getNoise()
        for i in range(OutL - r):
            dnoise = noise[i][0]
            n = i % r
            for j in range(c):
                TMP = WSave[n][j].copy()
                wb[i + r][j] = TMP + dnoise
            TMP = BSave[0][n].copy()
            bb[i + r] = TMP + dnoise
    
    print("=====4--------------")
    with torch.no_grad():
        wb, bb = kd.encryptKey(wb, bb)
        basenet.classifier[6].weight = nn.Parameter(wb)
        basenet.classifier[6].bias = nn.Parameter(bb)

    return basenet

def SplitModelVGG(basenet, net1, net2, net3):
    c = 4096
    OutL = 3000

    wbb = basenet.classifier[6].weight
    bbb = basenet.classifier[6].bias

    net1.classifier[6] = nn.Linear(in_features=c, out_features=OutL, bias=True)
    net2.classifier[6] = nn.Linear(in_features=c, out_features=OutL, bias=True)
    net3.classifier[6] = nn.Linear(in_features=c, out_features=OutL, bias=True)

    print("=====4--------------")
    with torch.no_grad():
        wb1, wb2, wb3 = torch.split(wbb, 1000, dim=0)
        bb1, bb2, bb3 = torch.split(bbb, 1000, dim=0)
        net1.classifier[6].weight = nn.Parameter(wb1)
        net1.classifier[6].bias = nn.Parameter(bb1)
        net2.classifier[6].weight = nn.Parameter(wb2)
        net2.classifier[6].bias = nn.Parameter(bb2)
        net3.classifier[6].weight = nn.Parameter(wb3)
        net3.classifier[6].bias = nn.Parameter(bb3)
    return net1, net2, net3

"""
wbt = np.ndarray((6,6))
for i in range (6):
    for j in range (6):
        wbt[i][j] = i + j

wbt_t = torch.from_numpy(wbt)
wbt1, wbt2, wbt3 = torch.split(wbt_t, 2, dim=0)

wbt_n = torch.concat((wbt1,wbt2,wbt3), dim=1)
wbt_nn = wbt_n.numpy()

n = 0
for i in range (6):
    for j in range (6):
        if (wbt[i][j] == wbt_nn[i][j]):
            n += 1
print(n)
"""
