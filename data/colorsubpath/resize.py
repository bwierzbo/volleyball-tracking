#!/usr/bin/python
from PIL import Image
import os
import sys
import numpy as np
import cv2 as cv

path = "/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/sort/ball//"
print(os.path.exists(path))
dirs = os.listdir( path )

size = 32
dim = 3

def rename():
   
    folder = "/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/sort/ball//"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"ball-{str(count)}.jpg"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            
            print("hello")
            pic = cv.imread(path+item)
            if pic is None:
                 print("none")
            else:
                img = cv.resize(pic, (size,size))
                img = np.reshape(img,[1,size,size,dim])
        else:
            break

            

rename()