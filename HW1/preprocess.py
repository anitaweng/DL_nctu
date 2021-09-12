#import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image

with open('test.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    a = []
    for row in rows:
        a.append(row)
    csvfile.close()

    label = np.zeros((394, 1))
for i in range(1, 395):
    if np.array(a[i])[3] == "bad":
        label[i - 1][0] = 0
    elif np.array(a[i])[3] == "good":
        label[i - 1][0] = 1
    else:
        label[i - 1][0] = 2

width = np.zeros((394, 1))
for i in range(1, 395):
    width[i - 1][0] = float(np.array(a[i])[1])

height = np.zeros((394, 1))
for i in range(1, 395):
    height[i - 1][0] = float(np.array(a[i])[2])

xmin = np.zeros((394, 1))
for i in range(1, 395):
    xmin[i - 1][0] = float(np.array(a[i])[4])

ymin = np.zeros((394, 1))
for i in range(1, 395):
    ymin[i - 1][0] = float(np.array(a[i])[5])

xmax = np.zeros((394, 1))
for i in range(1, 395):
    xmax[i - 1][0] = float(np.array(a[i])[6])

ymax = np.zeros((394, 1))
for i in range(1, 395):
    ymax[i - 1][0] = float(np.array(a[i])[7])

for i in range(1, 395):
    print(np.array(a[i])[0])
    img = mpimg.imread("images/" + np.array(a[i])[0])
    img_cropped = img[int(ymin[i-1][0]):int(ymax[i-1][0]), int(xmin[i-1][0]):int(xmax[i-1][0]), :]
    #fx = 128.0 / ((xmax[i - 1][0]) - (xmin[i - 1][0]))
    #fy = 128.0/((ymax[i-1][0])-(ymin[i-1][0]))
    #cv2.resize(img_cropped, (fx, fy))
    img_cropped = img_cropped.copy(order='C')
    mpimg.imsave("preprocess_test/" + str(i-1) + ".jpg", img_cropped)
    im = Image.open("preprocess_test/" + str(i-1) + ".jpg")
    width = 128
    height = 128
    nim2 = im.resize((width, height), Image.BILINEAR)
    nim2.save("preprocess_test/" + str(i-1) + ".jpg")
