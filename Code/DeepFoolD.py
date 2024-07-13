'''
Normal Deepfool 1/15/2021
Add return:
gradient
perturbation

work with the defence method. Num_classes should be doubled
'''


import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from Autograd import zero_gradients

import KeyDecrypt as kd


def deepfoolD(image, net1, net2, net3, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param basenet: base combined network
       :param net1, net2, net3: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net1 = net1.cuda()
        net2 = net2.cuda()
        net3 = net3.cuda()
    else:
        print("Using CPU")

    """
    f_image = torch.cat((f_image1, f_image2, f_image3),1)
    """

    x = Variable(image[None, :, :, :], requires_grad=True)

    f_image1 = net1.forward(x).data.cpu()
    f_image2 = net2.forward(x).data.cpu()
    f_image3 = net3.forward(x).data.cpu()

    I = f_image1.numpy().flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]
    
    f_image = torch.concat((f_image1, f_image2, f_image3), dim=1)
    """
    print(f_image1[0][0])
    print(f_image[0][0])
    print(f_image2[0][0])
    print(f_image[0][1000])
    print(f_image3[0][0])
    print(f_image[0][2000])
    """
    # Decrypt output
    f_image = kd.decryptKey(f_image)    

    #print(f_image1.shape)
    #print(f_image.shape)

    B= (f_image).numpy().flatten()[0:1000].argsort()[::-1]

    Originallabel = B[0]

    # Find pertubation, but we need a combined model to test

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    
    fsn1 = net1.forward(x)
    #fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label
    CountFlag=0
    while k_i == label and loop_i < max_iter:

        pert = np.inf

        fsn1[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        if CountFlag==0:
            TheGradient=x.grad.data.cpu().numpy().copy()
            CountFlag=1

        for k in range(1, num_classes):
            zero_gradients(x)

            fsn1[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fsn1[0, I[k]] - fsn1[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:

            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()

        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
        
        x = Variable(pert_image, requires_grad=True)
        fsn1 = net1.forward(x)
        k_i = np.argmax(fsn1.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot
    

    x = Variable(pert_image, requires_grad=True)
        
    fsn1 = net1.forward(x).data.cpu()
    fsn2 = net2.forward(x).data.cpu()
    fsn3 = net3.forward(x).data.cpu()

    k_i = np.argmax(fsn1.data.cpu().numpy().flatten())

    fsn = torch.concat((fsn1, fsn2, fsn3), dim=1)
    fsn = kd.decryptKey(fsn)
    Protected = np.argmax(fsn.numpy().flatten()[0:1000])

    return r_tot, loop_i, label, k_i, Originallabel,Protected,pert_image,TheGradient
