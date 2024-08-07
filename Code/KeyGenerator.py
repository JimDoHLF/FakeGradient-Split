import numpy as np
import random as random

# The purpose of this is to generate keys
# 1 key to encode, 1 key to decode
# Highest hexadecimal value (3 numbers) is 4095.
# Code currently built for 3000 values, split into 3 models with 1000 values each

def GenerateKey(size):
    # Key file will store the structure of the key
    keyFile = open("key.txt", "w")

    # Which position
    keyList = []
    for i in range (size):
        keyList.append(i)

    # Shuffle the keys
    random.shuffle(keyList)

    # Encode to hex and write
    for i in range (size):
        keyFile.write(hex(keyList[i])[2:].rjust(3, "0"))

    # Write key
    keyFile.close()

def GenerateNoise(size):
    noiseFile = open("noise.txt", "w")
    noise = random.sample(range(0,(size - 1000) * 2), size - 1000)
    random.shuffle(noise)

    # Encode to hex and write
    for i in range (size-1000):
        noiseFile.write(hex(noise[i])[2:].rjust(3,"0"))

    noiseFile.close()