import cv2
import numpy as np  
import pyautogui 

# Importing HaarCascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start Video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    row0, col0, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if (len(faces) == 0):                                               #moving to next frame if no face is being detected
        cv2.imshow('frame', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
        continue

    for (fx, fy, fw, fh) in faces:
        # cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        roi_gray = gray[fy:fy+int(fh/2), fx:fx+int(fw/2)]               # tracking only one eye (R) by sending half face
        roi_color = frame[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        if (len(eyes) == 0):                                            #moving to next frame if no eyes are detected
            cv2.imshow('frame', frame)
            key = cv2.waitKey(30)
            if key == 27:
                break
            continue
            
        for (ex, ey, ew, eh) in eyes:
            ey=ey+int(eh/4)                                             #removing eyebrows from roi
            eh=int(eh/2)            
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)
            frame1 = frame[faces[0][1]:faces[0][1]+faces[0]
                           [3], faces[0][0]:faces[0][0]+faces[0][2]]

            roi = frame1[eyes[0][1]+int(eyes[0][3]/4):(
                eyes[0][1]+int(3*eyes[0][3]/4)), eyes[0][0]:(eyes[0][0]+eyes[0][2])]
            
            if roi.shape[0]==0: 
                cv2.imshow('frame', frame)
                key = cv2.waitKey(30)
                if key == 27:
                    break
                continue
            # cv2.imshow('iris', roi)
            rows, cols, _ = roi.shape
            row1, col1, _ = frame1.shape
            # if(len(roi)==0): break
                
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
            # kernel=np.ones((5,5),np.uint8)
            # gray_roi=cv2.dilate(gray_roi,kernel,iterations=2)
            # gray_roi=cv2.erode(gray_roi,kernel,iterations=2)
            
            value = 10
            t_area =roi.shape[0] * roi.shape[1]                     #total ared of eye frame
            # Loop for the threshold value calculation
            while (value < 100):

                _, threshold = cv2.threshold(
                    gray_roi, value, 255, cv2.THRESH_BINARY_INV)    #thresholding the grayscale roi
                
                contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #finding contours in threshold
                # area=0
                
                contours = sorted(
                    contours, key=lambda x: cv2.contourArea(x), reverse=True)
                # print(contours)
                if(len(contours)>0):
                    area=cv2.contourArea(contours[0])               #area of max contour
                else:
                    value=value+1
                    continue
                if(area==0):
                    value=value+1
                    continue
                if(t_area/area<15):                                 #if max contour area is 15th of eye frame, identifying it as pupil
                    break
                value=value+1
                
            print(value)
            
            (x_axis,y_axis),radius = cv2.minEnclosingCircle(contours[0])
            center = (int(x_axis),int(y_axis))
            radius = int(radius)
            #pupil tracing by contour
            cv2.circle(roi,center,radius,(0,255,0),2)
            
            # area = cv2.contourArea(cnt)
            # if(area==0):continue
            b=t_area/area
            print (area,t_area,b)
            cv2.putText(frame,f'{b}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 2)
            cv2.imshow('Threshold',threshold)
            cv2.imshow('tracking_frame', frame)


    key = cv2.waitKey(30)
    if key == 27:
        break
cv2.destroyAllWindows()