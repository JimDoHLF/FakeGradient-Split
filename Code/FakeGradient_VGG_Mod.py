import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from imageio.v2 import imread

from ModifyModel import ModifyModelVGGScale
from ModifyModel import SplitModelVGG
from DeepFoolB import deepfoolB
from DeepFoolC import deepfoolC
from DeepFoolD import deepfoolD

import csv
import cv2

import KeyGenerator as kg

def runmodel(runtime):
    Scale = 20

    net = models.vgg19(weights='IMAGENET1K_V1').cuda()
    net.eval()

    fgnet_base = models.vgg19(weights='IMAGENET1K_V1')
    fgnet_base = ModifyModelVGGScale(fgnet_base, Scale).cuda()

    fgnet1 = models.vgg19(weights='IMAGENET1K_V1')
    fgnet2 = models.vgg19(weights='IMAGENET1K_V1')
    fgnet3 = models.vgg19(weights='IMAGENET1K_V1')
    fgnet1, fgnet2, fgnet3 = SplitModelVGG(fgnet_base, fgnet1, fgnet2, fgnet3)

    fgnet_base.eval()
    fgnet1.eval()
    fgnet2.eval()
    fgnet3.eval()

    #
    AT="DeepFool"
    CSVfilenameTime ='VGG19'+'_'+ AT +"_"+str(Scale)+"_MethodB"+'_Result.csv'
    fileobjT = open(CSVfilenameTime, 'w', newline='')  # 
    # fileobj.write('\xEF\xBB\xBF')#
    # 
    writerT = csv.writer(fileobjT)  # csv.writer(fileobj)writer writer
    ValueTime=['Original ATT,GT','Original ATT, ATT','On Fake ATT, GT','On Fake ATT,ATT','On Fake ATT, Def','ACC','ACC_ALL','DL2R','DL2G','DL2B','DLIR','DLIG','DLIB','AL2R','AL2G','AL2B','ALIR','ALIG','ALIB']
    writerT.writerow(ValueTime)
    CountT=0        #deepfool
    CountTotal=0    #
    CountDF_EFF=0   #deepfool 
    CountDF_EFF_Def=0  #DeepFool

    Folder='C:/Users/longd/Documents/ImageNet/ILSVRC/Data/DET/test/'
    FileName='ILSVRC2016_test'
    Append='.JPEG'            #00099990
    Error=[]
    for i in range(1,runtime): # Number of tries
        Index=str(i+1)
        K=len(Index)
        IndexFull='_'
        for j in range(8-K):
            IndexFull=IndexFull+str(0)
        IndexFull=IndexFull+Index
        FNAME=Folder+FileName+IndexFull+Append
        #im_orig = Image.open('test_im2.jpg')

        CC = cv2.imread(FNAME)
        #print(im_orig.size)
        a, b, c = CC.shape
        #print(CC.shape, c)

        image = imread(FNAME)
        if (len(image.shape) < 3):
            #print('gray')
            continue
        if c!=3:
            continue

        CountTotal=CountTotal+1

        #im_orig = Image.open('test_im2.jpg')
        #im_orig = Image.open('ILSVRC2012_test_00000002.JPEG')
        mean = [ 0.485, 0.456, 0.406 ]
        std = [ 0.229, 0.224, 0.225 ]
        #im_origB = Image.open('ILSVRC2012_test_00000002.JPEG')
        im_orig = Image.open(FNAME)
        im_origB = Image.open(FNAME)

        # Remove the mean
        im = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                std = std)])(im_orig)
        imB = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                std = std)])(im_origB)
        #r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)
        '''
        f_image = net.forward(Variable(im[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]
        Originallabel = I[0]
        '''
        #r, loop_i, label_orig, label_pert, Originallabel,Protected,pert_image,TheGradient = deepfoolC(im, fgnet_base)
        r, loop_i, label_orig, label_pert, Originallabel,Protected,pert_image,TheGradient = deepfoolD(im, fgnet1, fgnet2, fgnet3)
        rB, loop_iB, label_origB, label_pertB, pert_imageB,TheGradientB = deepfoolB(imB, net)
        #print("original:    ", Originallabel)
        #print("original:    ", Protected)
        #summary result
        print("Original Attack Result:  ", label_pertB, "    Original Label in original Attack: ",label_origB)
        print(" Attack Result In Fake:  ", label_pert, "   Original Label in Attack With Fake: ", Originallabel,"   Protected By Fake: ",Protected )
        Acc=0
        AccB=0
        if label_pert!=Protected:
            print("DeepFool Works!")
            CountDF_EFF=CountDF_EFF+1
            if label_origB==Protected:
                CountDF_EFF_Def=CountDF_EFF_Def+1
                Acc=1
        if label_origB == Protected:
            CountT=CountT+1
            AccB=1
        print("Efficiency: ===>", int(CountT*100/CountTotal))
        print()

    print("Final Result:   ",CountT,CountTotal,CountDF_EFF,CountDF_EFF_Def)
    ValueTime=[CountT,CountTotal,CountDF_EFF,CountDF_EFF_Def,int(CountT*100/CountTotal)]
    writerT.writerow(ValueTime)

    # Check value
    return (CountT * 100 / CountTotal)


# Run the model 10 times, get noise with highest efficiency
bestResult = 0
bestNoiseFile = open("bestnoise.txt", "w")
currentNoiseFile = open("noise.txt", "r")
for r in range (10):
    print("Pass: ", r)
    kg.GenerateNoise(3000)
    newResult = runmodel(100)
    if (newResult > bestResult):
        bestNoiseFile.write(currentNoiseFile.read())
        bestResult = newResult


bestNoiseFile.close()
currentNoiseFile.close()

print("Highest efficiency achieved: ", bestResult)