import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import glob

image_path = 'data/'
width = 32
height = 32
i = 0
image_list = []
for filename in glob.glob(image_path+'*.png'):
    img = Image.open(filename)
    new_img = img.resize((width, height), Image.BILINEAR)
    if i < 15000:
        new_img.save("preprocess_train/" + filename.split("/")[-1])
        print("preprocess_train/" + filename.split("/")[-1])
    else:
        new_img.save("preprocess_test/" + filename.split("/")[-1])
        print("preprocess_test/" + filename.split("/")[-1])
    i = i + 1
print(i)

