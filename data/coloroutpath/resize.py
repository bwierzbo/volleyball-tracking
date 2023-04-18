#!/usr/bin/python
from PIL import Image
import os
import sys

path = "/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/coloroutpath/ball//"
print(os.path.exists(path))
dirs = os.listdir( path )


def resize():
    for item in dirs:
        print("hello")
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            print(f)
            print("hello")
            newname = f.replace('c-','ball-')
            imResize = im.resize((32,32), Image.ANTIALIAS)
            imResize.save(newname + '.jpg', 'JPEG', quality=100)

            
resize()