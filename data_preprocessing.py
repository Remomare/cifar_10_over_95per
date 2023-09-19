import math
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys

def data_aguments(args, trainset):
    
    transform_dataset = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
                                            transforms.ToTensor()])
    transform_dataset = torchvision.datasets.ImageFolder(trainset, # 다운로드 받은 폴더의 root 경로를 지정합니다.
                                    transform=transform_dataset)

    # 데이터 로더를 생성합니다.
    transform_loader = torch.utils.data.DataLoader(transform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return transform_dataset, transform_loader

def data_aguments_use_opcv(args, trainset):
    arument_path = os.path.join(args.data_set , '/aguments')

    if not os.path.exists(arument_path):
        os.makedirs(arument_path)
    
    dataset_list = os.listdir(arument_path)
    trainset = main_TransformImage(args, dataset_list, arument_path)
    transform_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
 
    
    return trainset, transform_loader


def save(keyPath, file_name, cv_img, rate, type):
    
    if os.path.isdir(keyPath) != True:
        os.mkdir(keyPath)
    
    saved_name = os.path.join(keyPath,"{}{}.{}".format(file_name.split('.')[0], type, 'jpg'))
    print(saved_name)
    cv.imwrite(saved_name, cv_img)

def augmente(args, keyName, argument_path, rate=None, if_scale=False, ):
    
    saved_dir = argument_path
    keyPath = os.path.join(argument_path, keyName) # keypath direct to root path
    print(keyPath)
    datas = os.listdir(keyPath)
    data_total_num = len(datas)
    print("Overall data in {} Path :: {}".format(keyPath, data_total_num))
    
    
    try:
        for data in datas:
            type = "_scale_"
            data_path = os.path.join(keyPath, data)
            img = cv.imread(data_path)
            shape = img.shape
            ###### data rotate ######
            data_rotate(saved_dir, data, img, 20, "_rotate_", saving_enable=True)
            
            ###### data flip and save #####
            data_flip(saved_dir, data, img, rate, 1, False) # verical random flip
            data_flip(saved_dir, data, img, rate, 0, False) # horizen random flip
            data_flip(saved_dir, data, img, rate, -1, False) # both random flip
            
            ####### Image Scale #########
            if if_scale == True:
                print("Start Scale!")
                x = shape[0]
                y = shape[1]          
                f_x = x + (x * (rate / 100))
                f_y = y + (y * (rate / 100))
                cv.resize(img, None, fx=f_x, fy=f_y, interpolation = cv.INTER_CUBIC)

                img = img[0:y, 0:x]
                
                save(saved_dir, data, img, rate, type)
            ############################
                        
        #plt.imshow(img)
        #plt.show()
        return "success"
    
    except Exception as e:
        print(e)
        return "Failed"
    
def data_flip(saved_dir, data, img, rate, type, saving_enable=False):
    
    img = cv.flip(img, type)
    try:
        if type == 0:
            type = "_horizen_"
        elif type == 1:
            type = "_vertical_"
        elif type == -1:
            type = "_bothFlip_"
        
        if saving_enable == True:
            save(saved_dir, data, img, rate, type)
    
    except Exception as e:
        print(e)
        return "Failed"
    
def data_rotate(saved_dir, data, img, rate, type, saving_enable=False):
    
    xLength = img.shape[0]
    yLength = img.shape[1]
    
    try:
        rotation_matrix = cv.getRotationMatrix2D((xLength/2 , yLength/2), rate, 1)
        img = cv.warpAffine(img, rotation_matrix, (xLength, yLength))
        #print(img.shape)        
        if saving_enable == True:
            save(saved_dir, data, img, rate, type)
        
        return "Success"
    except Exception as e:
        print(e)
        return "Failed"
    
def main_TransformImage(args, keyNames, argument_path):
    try:
        for keyname in keyNames:
            
            #print(keyname)
            augmente(args, keyname, argument_path, 20 ) # scaling

        return "Augment Done!"
    except Exception as e:
        print(e)
        return "Augment Error!"