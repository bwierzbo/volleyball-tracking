#!/usr/bin/python
from PIL import Image
import os
import sys
import shutil

pathcolorball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/coloroutpath/ball/"
pathcolornotball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/coloroutpath/notball/"
pathmask = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/maskoutpath"
pathmaskball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/maskoutpath/ball/"
pathmasknotball = "C:/Users/Benjamin/OneDrive/Documents/VolleyballTracking/volleyball-tracking/data/maskoutpath/notball/"
print(os.path.exists(pathcolornotball))
#dirs = os.listdir( path )

if os.path.exists(pathmaskball):
    print("exists")
else:
    os.makedirs(pathmaskball)
    os.makedirs(pathmasknotball)
 


filelistcolor = []
filelistmask = []

for root, dirs, files in os.walk(pathcolornotball):
	for file in files:
        #append the file name to the list
		filelistcolor.append(os.path.join(root,file).split('c-')[1])

#now i have a list of the files numbers of not ball
for name in filelistcolor:
    print(name)

print(os.listdir(pathcolornotball))


# for root, dirs, files in os.walk(pathmask):
# 	for file in files:
		
# 		if os.path.join(root,file).split('b-')[1] in filelistcolor:
# 			shutil.move(file, pathmasknotball)
		
# 		filelistmask.append(os.path.join(root,file).split('b-')[1])

# #now i have a list of the files numbers of not ball
# for name in filelistmask:
#     print(name)






# def resize():
#     for item in dirs:
#         print("hello")
#         if os.path.isfile(path+item):
#             im = Image.open(path+item)
#             f, e = os.path.splitext(path+item)
#             print(f)
#             print("hello")
#             newname = f.replace('c-','ball-')
#             imResize = im.resize((32,32), Image.ANTIALIAS)
#             imResize.save(newname + '.jpg', 'JPEG', quality=100)

            
# resize()