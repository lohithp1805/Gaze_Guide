# time for patch detection(firtly eye ball is detected)
# minimum radius(limit after which mean is to be calculated)
# if merge happen one value preveious to that is to be used as threshold
# 2nd detection roundness limit(if after 3(tentative) frames there is huge change in roundness of Contour)
import cv2
import numpy as np
import random

er_ker = np.ones((20,20),np.uint8)
di_ker = np.ones((15,15),np.uint8)

def pupil(ip):
    rou_ar = []
    con_ar = []
    radius_ar = []
    for i in range(1,len(ip)):
        area = cv2.contourArea(ip[i])
        peri = cv2.arcLength(ip[i],True)
        hull = cv2.convexHull(ip[i])
        area_hull = cv2.contourArea(hull)
        rou = (4*np.pi*area)/(peri*peri) #roundness
        con = area/area_hull  #convexity
        (x,y),radius = cv2.minEnclosingCircle(ip[i])
        radius = int(radius)   #approx radius of contour
        rou_ar.append(rou)
        con_ar.append(con)
        radius_ar.append(radius)

    return rou_ar,con_ar,radius_ar

def contour(mat,mat2):
    final = []
    for i in range(len(mat)):
        final.append(mat2[mat[i]])
    if  len(mat)==1:
        return mat[final.index(np.max(final))]+1   
    else:
        return mat[final.index(np.max(final))]

def process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bf_img = cv2.bilateralFilter(image,25,75,75)
    erode_img = cv2.erode(bf_img,er_ker,iterations=1)
    di_img = cv2.dilate(erode_img,di_ker,iterations=1)
    return di_img

def thres_select(image):
    im_1 = process(image)
    a = 0
    b = 100    
    while(True):
        rani = random.randint(a,b)
        _ , thresh = cv2.threshold(im_1,rani,255,cv2.THRESH_BINARY)
        cont , _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _,_,c = pupil(cont)
        if len(c) == 0:
            a = rani
        elif np.max(c) > 50:
            b = rani
        elif np.max(c) < 20:
            continue
        else:
            break
    return a,b

img = cv2.imread("gaze11.jpeg")
start,stop = thres_select(img)

print(start, " " , stop)

for i in range(start,stop):
    img = cv2.imread("gaze11.jpeg")
    dil_img = process(img)
    _ , thresh = cv2.threshold(dil_img,i,255,cv2.THRESH_BINARY)
    _ , thresh1 = cv2.threshold(dil_img,i+1,255,cv2.THRESH_BINARY)
    cont , _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont1 , _ = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a,_,c = pupil(cont)
    _,_,c1 = pupil(cont1)
    a = np.array(a)
    if len(a)!=0 and len(c1)!=0:
        if np.max(c1) - np.max(c) >10:   # checking merger
            mat = np.where(a > 0.6)[0]
            # M = cont(contour(mat,c))
            # x = int(M['m10']/M['m00'])
            # y = int(M['m01']/M['m00'])
            cv2.drawContours(img, cont, contour(mat,c), (0,255,0), 3)
            cv2.imshow("frame",img)
            cv2.waitKey(0)
            break
    
        elif np.max(c1) > 50 and np.max(c1) < 100:
            mat = np.where(a > 0.6)[0]
            # M = cont(contour(mat,c))
            # x = int(M['m10']/M['m00'])
            # y = int(M['m01']/M['m00'])           
            cv2.drawContours(img, cont, contour(mat,c), (0,255,0), 3)
            cv2.imshow("frame",img)
            cv2.waitKey(0)
            break

# print(mat)
# cv2.drawContours(img, cont, contour(mat,c), (0,255,0), 3)
# cv2.imshow("frame",img)
# cv2.waitKey(0)
# img = cv2.imread("gaze16.jpeg",0)
# img = cv2.convertScaleAbs(img, 0.2)
# bf_img = cv2.bilateralFilter(img,25,75,75)
# erode_img = cv2.erode(bf_img,er_ker,iterations=1)
# di_img = cv2.dilate(erode_img,di_ker,iterations=1)
# cv2.imshow("frame",di_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
