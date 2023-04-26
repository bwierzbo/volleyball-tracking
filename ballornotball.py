
#import os for file writing
#importing openCV full package - pip install opencv-contrib-python
#importing numpy for math
import os
import sys
import cv2 as cv
import numpy as np
import model_loader as ml


testpath = ('../volleyball-tracking/data/testpath')

videoCapture = cv.VideoCapture('../volleyball-tracking/volleyballVideos/theclip.mp4')
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

backSub = cv.createBackgroundSubtractorKNN()
n=0

while True:
    ret, frame = videoCapture.read()
    if not ret: break

    mask = backSub.apply(frame)

    mask = cv.GaussianBlur(mask, (13, 13),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT_ALT, 1.5, 5, param1=300, param2= 0.85, minRadius=4, maxRadius=22)


    if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None
            for i in circles[0, :]:
                if chosen is None: chosen = i
                if prevCircle is not None:
                    if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                        chosen = i

            #using the radius to get the starting x,y coordinates and the width and height for the image cut
            rx = chosen[0] - chosen[2]
            ry = chosen[1] + chosen[2]
            rw = chosen[2]*2
            rh = chosen[2]*2

            #cutting images from black and white mask
            cut_m = mask[ry - rh - 5 : ry + 5, rx - 5 : rx + rw + 5]

            #cutting images from color mask
            cut_f = frame[ry - rh - 5 : ry + 5, rx - 5 : rx + rw + 5]
            
            #Splice cut images to cutout background of circle detection
            cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)

            #Checking images with new model
            if cut_c is None:
                 print("none")
            else:
                if ml.checkIMG(cut_c) == 1:
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,252,124), 3)
                else:
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,0,255), 3)

            n+=1
            prevCircle = chosen

    cv.imshow("ModelPredict", frame)

    if cv.waitKey(10) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()