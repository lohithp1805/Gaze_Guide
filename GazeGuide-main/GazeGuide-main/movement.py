import cv2
import numpy as np
import pyautogui as p
import collections

p.FAILSAFE = False
# Importing HaarCascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

screen_w,screen_h = p.size()

buffer_size = 10
position_buffer = collections.deque(maxlen=buffer_size)
buffer_size2 = 10
cab_buffer = collections.deque(maxlen=buffer_size2)

# initialize video capture device
cap = cv2.VideoCapture(0)
img  = np.zeros((screen_h,screen_w,3),np.uint8)
c=0
color_list = [
    (0,255,0),(0,235,0),(0,215,0),(0,195,0),(0,175,0),(0,155,0),(0,135,0),(0,115,0),(0,95,0),(0,75,0)
]
avg_x,avg_y = 0,0
# define calibration points on screen
# in this example, we define 9 calibration points in a 3x3 grid
# calibration_points = np.array([
#     (100, 100), (200, 100), (300, 100),
#     (100, 200), (200, 200), (300, 200),
#     (100, 300), (200, 300), (300, 300)
# ], dtype=np.float32)

# load the Homography transformation matrix calculated during calibration
H = np.load('H_matrix2.npy')

# define the size of the screen
SCREEN_SIZE = (screen_w, screen_h)

# # define the region of the screen where the cursor will be controlled
REGION_SIZE = (SCREEN_SIZE[0]//2, SCREEN_SIZE[1]//2)
REGION_CENTER = (SCREEN_SIZE[0]//4, SCREEN_SIZE[1]//4)

# # define the boundaries of the region
REGION_LEFT = REGION_CENTER[0] - REGION_SIZE[0]//2
REGION_TOP = REGION_CENTER[1] - REGION_SIZE[1]//2
REGION_RIGHT = REGION_CENTER[0] + REGION_SIZE[0]//2
REGION_BOTTOM = REGION_CENTER[1] + REGION_SIZE[1]//2

while True:
        ret, frame = cap.read()
        if ret is False:
            break
        row0, col0, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
           
        if (len(faces) == 0):
            cv2.imshow('frame', frame)

            key = cv2.waitKey(30)
            if key == ord('q'):
                break
            continue

        for (fx, fy, fw, fh) in faces:
            # cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            
            roi_gray = gray[fy:fy+int(fh/2), fx:fx+int(fw/2)]
            roi_color = frame[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            if (len(eyes) == 0):
                cv2.imshow('frame', frame)
                
                key = cv2.waitKey(1)
                if key != -1:
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
                        roi = frame1[eyes[0][1]+int(eyes[0][3]/4):(
                            eyes[0][1]+int(3*eyes[0][3]/4)), eyes[0][0]:(eyes[0][0]+eyes[0][2])]
                        if roi.shape[0]==0: 
                            cv2.imshow('frame', frame)
                            key = cv2.waitKey(1)
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
                        t_area =roi.shape[0] * roi.shape[1]
                        c+=1 
                        if(c>10):
                         img.fill(0)
                         c=0

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
                    
                        if(len(contours)==0):continue
                        (x_axis,y_axis),radius = cv2.minEnclosingCircle(contours[0])

                        center = (int(x_axis),int(y_axis))    
                        position_buffer.append(center)
                        radius = int(radius)

                        cv2.circle(roi,center,radius,(0,255,0),2)   
                        
                        avg_x = sum([pos[0] for pos in position_buffer])/len(position_buffer)
                        avg_y = sum([pos[1] for pos in position_buffer])/len(position_buffer)
                                                    
                        pupil_x, pupil_y = avg_x,avg_y

    # map pupil position to corresponding position on calibration screen using the mapping from calibration
                        homog = np.array([pupil_x,pupil_y,1])
                        # homog = np.array([norm[0],norm[1],1]) 
            # map pupil position to corresponding position on calibration screen using the mapping from calibration
                        calibrated_position = np.dot(H, homog)
                        calibrated_position = (calibrated_position[0]/calibrated_position[2], calibrated_position[1]/calibrated_position[2])
                        #print(calibrated_position)

            # move the cursor to the calibrated position
                        cab_x1,cab_y1 = calibrated_position[0],calibrated_position[1]
                        cab_buffer.append([cab_x1,cab_y1])
                        x_min = min(cab_buffer,key = lambda p : p[0])[0]
                        x_max = max(cab_buffer,key = lambda p : p[0])[0]
                        y_min = min(cab_buffer,key = lambda p : p[1])[1]
                        y_max =max(cab_buffer,key = lambda p : p[1])[1]
                        gaze_points = np.array(cab_buffer)
                        center = np.mean(cab_buffer,axis=0)
                        print(center)
                        distances = np.linalg.norm(gaze_points-center,axis=1)

                        closest_index = np.argmin(distances)
                        central_gaze = gaze_points[closest_index]

                        cab_x1,cab_y1 = central_gaze[0],central_gaze[1]

                        cv2.circle(img,(int(cab_x1),int(cab_y1)),100,color_list[9-c],-1)                        

                        if(-32765<=cab_x1<=32765) and (-32765<=cab_y1<=32765): #limits are (-2^15-2, 2^15-1) and 2^15 = 32768
                        # struct.error: short format requires (-0x7fff - 1) <= number <= 0x7fff 
                          p.moveTo(cab_x1, cab_y1,duration=0)
                        

                        # display calibrated pupil position on screen
                        cv2.circle(frame, (int(cab_x1),int(cab_y1)), 10, (0,0,255), thickness=-1)
                        cv2.namedWindow("white screen", cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty ("white screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow('white screen',img)

                        # display video frame
                        cv2.imshow('frame', frame)

    # quit program if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
         break

# release video capture device and close all windows
cap.release()
cv2.destroyAllWindows()