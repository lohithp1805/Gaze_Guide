import cv2
import numpy as np
import pyautogui as p
import collections
p.FAILSAFE = False

# initialize video capture device
cap = cv2.VideoCapture(0)

# importing HaarCascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

buffer_size = 10
position_buffer = collections.deque(maxlen=buffer_size)

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = 0.03 * np.eye(4,dtype = np.float32) 
kalman.measurementNoiseCov = 0.01 *np.eye(2,dtype = np.float32) 
r = 0

screen_w,screen_h = p.size()
img  = np.zeros((screen_h,screen_w,3),np.uint8)

# choosen points for calibration
calibration_points = np.array([
    (0,0),(screen_w//4,0),(screen_w//2,0),(3*screen_w//4,0),(screen_w,0),(screen_w,screen_h//2),(screen_w,screen_h-100),(3*screen_w//4,screen_h-100),
    (screen_w//2,screen_h-100),(screen_w//4,screen_h-100),(0,screen_h-100),(0,screen_h//2),(screen_w//4,screen_h//4),(screen_w//2,screen_h//2),(3*screen_w//4,screen_h//4)

], dtype=np.float32)


# initialize empty list to store eye position data captured during calibration
eye_positions = []

# loop for capturing eye position data
for i in range(calibration_points.shape[0]):
    # show current calibration point on screen
    (x00,y00) = calibration_points[i]
    print(x00,y00)
    cv2.rectangle(img,(0,0),(screen_w,screen_h),(255,255,255),-1)
    cv2.circle(img,(int(x00)+5,int(y00)-91),50,(0,255,0),-1)
    position_buffer.clear()
    # wait for user to focus on calibration point and press any key
    while True:
        ret, frame = cap.read()
        p.moveTo(x00,y00,duration=0)
        if ret is False:
            break
        row0, col0, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if (len(faces) == 0):
            #cv2.imshow('frame', frame)
            key = cv2.waitKey(30)
            if key == 27:
                break
            continue

        for (fx, fy, fw, fh) in faces:
            # cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            roi_gray = gray[fy:fy+fh, fx:fx+int(fw/2)]
            roi_color = frame[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            if (len(eyes) == 0):
                # cv2.imshow('frame', frame)
                key = cv2.waitKey(30)
                if key == 27:
                    break
                continue
                
            for (ex, ey, ew, eh) in eyes:
                ey=ey+int(eh/4)
                eh=int(eh/2)            
                cv2.rectangle(roi_color, (ex, ey),
                            (ex + ew, ey + eh), (0, 255, 0), 2)
                frame1 = frame[faces[0][1]:faces[0][1]+faces[0]
                            [3], faces[0][0]:faces[0][0]+faces[0][2]]
                eyes[0][3],eyes[0][2]=int(eyes[0][3]/3)*3,int(eyes[0][2]/3)*3
                eye_orig_image = frame1[eyes[0][1]+int(eyes[0][3]/4):(
                    eyes[0][1]+int(3*eyes[0][3]/4)), eyes[0][0]:(eyes[0][0]+eyes[0][2])]
                if eye_orig_image.shape[0]==0: 
                    # cv2.imshow('frame', frame)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    continue
                # cv2.imshow('iris', eye_orig_image)
        
                roi = eye_orig_image
                rows, cols, _ = roi.shape
                row1, col1, _ = frame1.shape
                # if(len(roi)==0): break
                    
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                # kernel=np.ones((5,5),np.uint8)
                # gray_roi=cv2.dilate(gray_roi,kernel,iterations=2) 
                # gray_roi=cv2.erode(gray_roi,kernel,iterations=2) 
                value = 10
                t_area =roi.shape[0] * roi.shape[1]

                while (value < 100):

                    _, threshold = cv2.threshold(
                        gray_roi, value, 255, cv2.THRESH_BINARY_INV)
                    
                    contours, _ = cv2.findContours(
                    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # area=0
                    contours = sorted(
                        contours, key=lambda x: cv2.contourArea(x), reverse=True)
                    # print(contours)
                    if(len(contours)>0):area=cv2.contourArea(contours[0])
                    else:
                        value=value+1
                        continue
                    if(area==0):
                        value=value+1
                        continue
                    if(t_area/area<20):
                        break
                    value=value+1
                print(value)

                for cnt in contours: 
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    cv2.line(frame, (fx+ex+x + int(w/2), 0),
                            (fx+ex+x + int(w/2), row0), (0, 255, 0), 2)
                    cv2.line(frame, (0, fy+ey+y + int(h/2)),
                            (col0, fy+ey+y + int(h/2)), (0, 255, 0), 2)   

                    (x_axis,y_axis),radius = cv2.minEnclosingCircle(contours[0])

                    center = (int(x_axis),int(y_axis))
                    radius = int(radius)    
                    position_buffer.append(center)
                    cv2.circle(roi,center,radius,(0,255,0),2)   
            
                    avg_x = sum([pos[0] for pos in position_buffer])/len(position_buffer)
                    avg_y = sum([pos[1] for pos in position_buffer])/len(position_buffer)

                    # cv2.imshow("Threshold", threshold)
                    # cv2.imshow("gray roi", gray_roi)
                    # cv2.imshow("Roi", roi)
                    cv2.imshow('Calibration Point', img)
                    break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('p'):
            exit()
        if key !=-1 :
            break

    eye_positions.append((avg_x, avg_y))    


# convert list of eye positions to numpy array
eye_positions = np.array(eye_positions, dtype=np.float32)
print(eye_positions)

# perform perspective transform to calculate mapping between eye positions and calibration points
H, _ = cv2.findHomography(eye_positions, calibration_points)
np.save('H_matrix2q.npy',H)
# release video capture device and close all windows
cap.release()
cv2.destroyAllWindows()                           