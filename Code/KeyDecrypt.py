import numpy as np
import torch
from KeyGenerator import GenerateKey

FOOLGRAD = 3000

def getNoise():
    noiseFile = open("noise.txt", "r")
    noiseStr = noiseFile.read()
    noise = np.ndarray(shape=(FOOLGRAD-1000,1), dtype=np.float32)
    for i in range (FOOLGRAD-1000):
        # Convert from hex to number
        temp = int(noiseStr[i * 3 : i * 3 + 3], 16)
        noise[i][0] = temp / 27500 # Modulate noise intensity
    return torch.from_numpy(noise)


def getKey32():
    keyFile = open("key.txt", "r")
    keyStr = keyFile.read()
    key = np.ndarray(shape=(FOOLGRAD,FOOLGRAD), dtype=np.float32)
    for i in range (FOOLGRAD):
        # Convert from hex to number
        temp = int(keyStr[i * 3 : i * 3 + 3], 16)
        key[i][temp] = 1
    return torch.from_numpy(key)

def encryptKey(weight, bias):
    key32 = torch.t(getKey32())
    print("Encrypting Model:")
    print(weight.shape)
    print(bias.shape)
    w = torch.matmul(key32,weight)
    b = torch.matmul(key32,bias)
    print("Model Encrypted")
    print(w.shape)
    print(b.shape)
    return w, b

def decryptKey(output):
    key32 = getKey32()
    TMP = torch.t(output)
    output = torch.matmul(key32,TMP)
    return output

GenerateKey(FOOLGRAD)