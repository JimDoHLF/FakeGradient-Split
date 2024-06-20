import numpy as np
import torch

def getKey32():
    keyFile = open("key.txt", "r")
    keyStr = keyFile.read()
    key = np.ndarray(shape=(4095,4095), dtype=np.float32)
    for i in range (4095):
        # Convert from hex to number
        temp = int(keyStr[i * 3 : i * 3 + 3], 16)
        key[i][temp] = 1
    return torch.from_numpy(key)

def encryptKey(weight, bias):
    key32 = torch.t(getKey32())
    w = torch.matmul(key32,weight)
    b = torch.matmul(key32,bias)
    return w, b

def decryptKey(output):
    key32 = getKey32()
    TMP = torch.t(output)
    ouptut = torch.matmul(key32,TMP)
    return output