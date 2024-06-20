import numpy as np
import torch
import torch.nn as nn

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

    OutL=4095 # 3 character Hex

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
        for i in range(OutL - r):
            n = i % r
            for j in range(c):
                TMP = WSave[n][j].copy()
                wb[i + r][j] = TMP
            TMP = BSave[0][n].copy()
            bb[i + r] = TMP

    
    print("=====4--------------")
    with torch.no_grad():
        wb, bb = kd.encryptKey(wb, bb)
    return basenet
