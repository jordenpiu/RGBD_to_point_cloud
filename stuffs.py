import torch 
from torchvision.datasets import Cityscapes
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import torch.nn as nn 
import argparse
from arg import args


dataset = Cityscapes(args.root, "train", "fine", "semantic")

print(dataset[0][0].size)

fig, ax = plt.subplots(ncols=2,figsize=(12,8))
ax[0].imshow(dataset[0][0])
ax[1].imshow(dataset[0][1], cmap='gray')
plt.show()

#label understanding reference 
#https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
#https://github.com/meetps/pytorch-semseg/tree/master/ptsemseg

ignore_index = 255
void_classes = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,-1]
valid_classes = [ignore_index,7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',\
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
                'train', 'motorcycle', 'bicycle']

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)
print(class_map)

colors = [ [ 0,0,0],
          [128,64,128],
          [224,35,232],
          [70,70,70],
          [102,102,156],
          [190,153,153],
          [153,153,153],
          [250,170,30],
          [220,220,0],
          [107,142,35],
          [152,251,152],
          [0,130,180],
          [220,20,60],
          [225,0,0],
          [0,0,142],
          [0,0,70],
          [0,60,100],
          [0,80,100],
          [0,0,230],
          [119,11,32]]
label_colors = dict(zip(range(n_classes),colors))

def encode_segmap(mask):
    for voidc in void_classes:
        mask[mask==voidc] = ignore_index
    for validc in valid_classes:
        mask[mask == validc]  = class_map[validc]
    return mask

def decode_segmap(temp):
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0]
        g[temp == l] = label_colors[l][1]
        b[temp == l] = label_colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1],3))
