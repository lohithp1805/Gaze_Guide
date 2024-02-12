import cv2
import numpy as np
import pyautogui as p
import collections

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

screen_w, screen_h = p.size()
p.FAILSAFE = False
a = np.array([screen_w/2, screen_h/2])

buffer_size = 10
position_buffer = collections.deque(maxlen=buffer_size)
buffer2 = collections.deque(maxlen=buffer_size)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if ret is False:
        break

    row0, col0, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if (len(faces) == 0):
        cv2.imshow('frame', frame)

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
            cv2.imshow('frame', frame)

            key = cv2.waitKey(30)
            if key == 27:
                break
            continue
        # print(eyes[0][0],eyes[0][1],eyes[0][2],eyes[0][3])

        for (ex, ey, ew, eh) in eyes:
            ey = ey+int(eh/4)
            eh = int(eh/2)
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

            frame1 = frame[faces[0][1]:faces[0][1]+faces[0]
                           [3], faces[0][0]:faces[0][0]+faces[0][2]]

            eye_orig_image = frame1[eyes[0][1]:(
                eyes[0][1]+eyes[0][3]), eyes[0][0]:(eyes[0][0]+eyes[0][2])]

            cv2.imshow('iris', eye_orig_image)

            roi = eye_orig_image
            rows, cols, _ = roi.shape
            row1, col1, _ = frame1.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            value = 0

            _, threshold = cv2.threshold(
                gray_roi, 100, 255, cv2.THRESH_BINARY_INV)
            contour, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            
            area = 0
            for cnt in contour:
                area = area+cv2.contourArea(cnt)

            while (value < 100):

                _, threshold = cv2.threshold(
                    gray_roi, value, 255, cv2.THRESH_BINARY_INV)
                th, tw = threshold.shape
                split1 = threshold[0:int(th/2), 0:tw]
                split2 = threshold[int(th/2):th, 0:tw]
                contour1, _ = cv2.findContours(
                    split1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                contour2, _ = cv2.findContours(
                    split2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                area1 = 0
                area2 = 0
                for cnt in contour1:
                    area1 = area1+cv2.contourArea(cnt)
                for cnt in contour2:
                    area2 = area2+cv2.contourArea(cnt)

                if (area1 + area2 > 0.6*area):
                    value = 40
                    break
                elif (area1 > 0.8*area2):
                    break

                value = value+1

            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if (len(contours) > 0):
                pupil_contour = max(contours, key=cv2.contourArea)

                # Find the controid of the pupil contour
                M = cv2.moments(pupil_contour)
                pupil_cx = float(M['m10']/M['m00'])
                pupil_cy = float(M['m01']/M['m00'])
                pupil_cx1 = int(M['m10']/M['m00'])
                pupil_cy1 = int(M['m01']/M['m00'])
                #print(pupil_cx,pupil_cy,sep="  ")
                # Draw a circle at the centroid of the pupil contour
                cv2.circle(eye_orig_image, (pupil_cx1, pupil_cy1),
                           5, (0, 255, 0), -1)

                # Estimate the gaze direction based on the positiion of the pupil
                eye_center_x = ew/2
                eye_center_y = eh/2

                #current_x ,current_y= p.position()

                x_off = pupil_cx-eye_center_x
                y_off = pupil_cy-eye_center_y

                #    if abs(x_off)>3 or abs(y_off)>3:
                if x_off >= 0:
                    new_x = screen_w//2 + x_off*70
                else:
                    new_x = screen_w//2 + x_off*100
                new_y = screen_h//2 + y_off*90
                position_buffer.append((new_x, new_y))
                buffer2.append((x_off, y_off))
                avg_x = sum([pos[0] for pos in position_buffer]) / \
                    len(position_buffer)
                avg_y = sum([pos[1] for pos in position_buffer]) / \
                    len(position_buffer)
                avg_x1 = sum([pos[0] for pos in buffer2])/len(buffer2)
                avg_y1 = sum([pos[1] for pos in buffer2])/len(buffer2)
                print(avg_x1, avg_y1, sep=" ")
                #print(eye_center_x,eye_center_y,sep=" ")
                #    gaze_x = eye_center_x + (pupil_cx-ew//2)*3
                #    gaze_y = eye_center_y + (pupil_cy - eh//2)*4
                #print(gaze_x,gaze_y,sep="  ")
                if 0 <= avg_x <= screen_w and 0 <= avg_y <= screen_h:
                    if (9 > abs(avg_x1) or 9 > abs(avg_y1)):
                        if (avg_x1 < 2 and avg_y1 > -1.7):
                            p.moveRel(-50, 0, duration=0)
                        elif (avg_x1 < 2 and avg_y1 < -1.7):
                            p.moveRel(-50, -50, duration=0)
                        elif (avg_x1 > 3 and avg_y1 < -1.7):
                            p.moveRel(50, -50, duration=0)
                        elif (avg_x1 > 3 and avg_y1 > -1.7):
                            p.moveRel(50, 0, duration=0)

            # cv2.imshow("Threshold", threshold)
            # cv2.imshow("gray roi", gray_roi)
            # cv2.imshow("Roi", roi)
            cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
