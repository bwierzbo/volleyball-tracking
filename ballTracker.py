import math
import cv2 as cv
import numpy as np
import model_loader as ml
import sys

#detect circle add it to class predict if is is a ball 
#if it is predicted as ball need to keep track of if ball gets tracked in specified permimeter
#if 2-6 balls are detected in range of eachother with a trajectory this is an important trajectory to keep track of
#if there are no balls detected after in range of eachother then this is likely a false prediction 

circle = 0

R=60

STATUS_INITIALIZE = 0
STATUS_STATIC = 1
STATUS_DYNAMIC = 2






def dist(x1, y1, x2, y2):
  dx = x1 - x2
  dy = y1 - y2
  return math.sqrt(dx * dx + dy * dy)

class Circle:
    #a class for all detected circles
    count = 1

    #attribute
    attr1="circle"
    def __init__(self, x, y, r, a):
        self.id = Circle.count
        Circle.count += 1
        self.pts = [[x, y]]
        self.pp = [[r, a]]
        self.status = STATUS_INITIALIZE
        self.v = None
        self.age = a
        self.nx = None
        self.ny = None

    def fit(self, x, y, r):
        d = dist(self.pts[-1][0], self.pts[-1][1], x, y)
        return d < R, d
    
    def add(self, x, y, r, a):
        self.pts.append([x, y])
        self.pp.append([r, a])
        self.age = a
        if len(self.pts) > 2: #need at least three points to set status
            #if self.status == STATUS_DIRECTED and self.nx is not None:
            #   print("Predict", self.nx, self.ny, "vs", x, y)
            #looking at the distances between detected circles
            dx1 = self.pts[-2][0] - self.pts[-3][0]
            dy1 = self.pts[-2][1] - self.pts[-3][1]

            dx2 = x - self.pts[-2][0]
            dy2 = y - self.pts[-2][1]

            d1 = dist(self.pts[-2][0], self.pts[-2][1], x, y)
            d2 = dist(self.pts[-2][0], self.pts[-2][1], self.pts[-3][0], self.pts[-3][1])
            #IF calculated distance is greater than a specific range then the detected circle is assigned a dynamic status 
            if dx1 * dx2 > 0 and dy1 * dy2 > 0 and d1 > 5 and d2 > 5:
                self.status = STATUS_DYNAMIC
                #print("Directed", self.pts)
                #self.predict()
            elif self.status != STATUS_DYNAMIC:
                self.status = STATUS_STATIC



    def predict(self):
        npts = np.array(self.pts)
        l = len(self.pts) + 1
        idx = np.array(range(1, l))

        kx = np.polyfit(idx, npts[:,0], 1)
        fkx = np.poly1d(kx)

        ky = np.polyfit(idx, npts[:,1], 1)
        fky = np.poly1d(ky)

        self.nx = fkx(l)
        self.ny = fky(l)
        return self.nx, self.ny
    
#array of circle
#dc detected circle, prev_dc previous detected circle
C=[]
dc = None
prev_dc = None

def get_circle():
    return dc


def find_fcircle(x, y, r):
    global C, circle
    rbp = []
    sbp = []

    for c in C:
        ft, d = c.fit(x, y, r)
        if ft:
            if circle - c.age < 4:
                rbp.append([c, d])
            elif c.status == STATUS_STATIC:
                sbp.append([c, d])
    
    if len(sbp) + len(rbp) == 0:
        return None
    rbp.sort(key = lambda e: e[1])
    if len(rbp) > 0:
        return rbp[0][0]
    
    sbp.sort(key = lambda e: e[1])
    return sbp[0][0]


def handle_circle(x, y, r):
    global C, circle, dc
    c = find_fcircle(x, y, r)
    if c is None:
        C.append(Circle(x, y, r, circle))
        return
    c.add(x, y, r, circle)
    if c.status == STATUS_DYNAMIC:
        if dc is None:
            dc=c 
        elif len(c.pts) > len(dc.pts):
            dc=c 

def begin_gen():
  global dc, prev_dc
  prev_dc = dc
  dc = None

def end_gen():
  global circle, dc
  circle += 1    

def handle_circles(mask, frame):


    prevCircle = None
    c = 0

    # see for HoughCircles perameter description https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT_ALT, 1.5, 90, param1=300, param2= 0.85, minRadius=4, maxRadius=19)
    
    begin_gen()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i
            #cv.circle(frame, (chosen[0], chosen[1]), 1, (0,100,100), 3)
            #cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)

        #chosen 2 is radius 

        #using the radius to get the starting x,y coordinates and the width and height for the image cut
        rx = chosen[0] - chosen[2]
        ry = chosen[1] + chosen[2]
        rw = chosen[2]*2
        rh = chosen[2]*2
            
            #rectangle shows what it is cutting for the classifier
            #cv.rectangle(frame,((rx-5), ry+5), ((rx+rw+5),(ry-rh-5)),(255,0,255), 3)
            #print(chosen[2])


            #cutting images from black and white mask
        cut_m = mask[ry - rh - 5 : ry + 5, rx - 5 : rx + rw + 5]
            #cv.imwrite("{0}/b-{1:03d}.jpg".format(mask_out_path, c), cut_m)

            #cutting images from color mask
            #cut_f = frame[ry - rh - 5 : ry + 5, rx - 5 : rx + rw + 5]
            #cv.imwrite("{0}/c-{1:03d}.jpg".format(color_out_path, c), cut_f)

        c+=1
        prevCircle = chosen

        end_gen()
        



def draw_ball(pic):
    bb = get_circle()
    #if not bb is None:
    #cv.circle
    #else:
    #if prev_bb is not None:
    #x,y = prev_bb.predict()
    #cv.circle

def draw_ball_path(pic):
    bb = get_circle()



def draw_circles(w, h):
    pic = np.zeros((h, w, 3), np.uint8)
    #draw_ball(pic)
    return pic

















def test_clip(path):
    videoCapture = cv.VideoCapture(path)
    videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)

    backSub = cv.createBackgroundSubtractorKNN()

    n=0

    while True:
        ret, frame = videoCapture.read()
        if not ret: break

        mask = backSub.apply(frame)

        mask = cv.GaussianBlur(mask, (13, 13),0)
        ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)


        #function to detect circles
        handle_circles(mask, frame)
        #function to draw circles




if __name__ == "__main__":
    test_clip(sys.argv[1])
    #test_clip("../volleyball-tracking/volleyballVideos/testingball.mp4")