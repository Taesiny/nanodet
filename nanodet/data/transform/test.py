# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 05:15:36 2021

@author: zm3171
"""

import cv2
import os,os.path


path1= 'C:/Users/zm3171/Desktop/test/211102_1/data/'
path2= 'C:/Users/zm3171/Desktop/test_reshape/211102_1/data/'

img_names=[]
imgs=[]
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path1):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img_names.append(f)
    
for image in img_names:
    img=cv2.imread(os.path.join(path1,image),cv2.IMREAD_GRAYSCALE)
    img1=cv2.vconcat([img[:,:256],img[:,256:512],img[:,512:768],img[:,768:]])
    cv2.imwrite(os.path.join(path2 , image), img1)
