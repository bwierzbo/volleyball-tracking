#!/usr/bin/python
from PIL import Image
import os
import sys
import shutil

pathcolorball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/coloroutpath/ball/"
pathcolornotball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/coloroutpath/notball/"
pathmask = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/maskoutpath/"
pathmaskball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/maskoutpath/ball/"
pathmasknotball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/maskoutpath/notball/"
colorsubpath = "C:/Users/Benjamin/OneDrive/Documents/volleyball-tracking/data/colorsubpath"
print(os.path.exists(colorsubpath))
dirs = os.listdir(colorsubpath)






def resize():
    for item in dirs:
        print("hello")
        if os.path.isfile(colorsubpath+item):
            im = Image.open(colorsubpath+item)
            f, e = os.path.splitext(colorsubpath+item)
            print(f)
            print("hello")
            newname = f.replace('c-','ball-')
            imResize = im.resize((32,32), Image.ANTIALIAS)
            imResize.save(newname + '.jpg', 'JPEG', quality=100)

            
resize()