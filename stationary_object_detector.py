import cv2
import mediapipe as mp
import numpy as np
import pybgs as bgs
import os
import math

a = 0
count = 1
folder_num = 1
centre_1 = None
centre_2 = None
centre_3 = None
min_area_threshold = 6000
video_path = "final_video.mp4"
roi = []
time_stamps = {}

def get_coords(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at coordinates (", x, ", ", y, ")")
        if len(roi) < 2:
            roi.append([x,y])

def draw_grids(frame, grid_size, color, thickness):
    for i in range(0, frame.shape[1], grid_size):
        cv2.rectangle(frame, (i, 0), (i+grid_size, frame.shape[0]), color, thickness)   
    for j in range(0, frame.shape[0], grid_size):
        cv2.rectangle(frame, (0, j), (frame.shape[1], j+grid_size), color, thickness)
    return frame

cv2.namedWindow("video")
cv2.setMouseCallback("video", get_coords)

algorithm = bgs.DPAdaptiveMedian()
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    pos_frame = cap.get(1)
    timestamp = pos_frame/fps
    frame_copy = frame.copy()
    frame = draw_grids(frame, grid_size = 50, color = (0,255,0), thickness = 1)
    img_output = algorithm.apply(frame_copy)
    img_bgmodel = algorithm.getBackgroundModel()
    rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    hand_detector = mp_hands.process(rgb_frame)
    if hand_detector.multi_hand_landmarks:
        hand_detector_flag = True
    else:
        hand_detector_flag = False

    if len(roi) >= 2:
        filtered_contours = []
        img_output = img_output[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        num_white_pixels = cv2.countNonZero(img_output)

        if num_white_pixels > 1700:
            _, contours,_ = cv2.findContours(img_output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area_threshold: 
                    filtered_contours.append(contour)
            if (len(filtered_contours) == 1):
                M = cv2.moments(filtered_contours[0])
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                if hand_detector_flag == False:
                    if count == 1:
                        centre_1 = [cx,cy]
                        count += 1
                    elif count == 2:
                        centre_2 = [cx,cy]
                        if centre_1 == centre_2:
                            count += 1
                        else:
                            count = 1
                            centre_1 = None
                            centre_2 = None
                            centre_3 = None
                    elif count == 3:
                        centre_3 = [cx, cy]
                        if abs(math.dist(centre_2, centre_3)) <= 2:
                            a += 1
                        elif abs(math.dist(centre_2, centre_3)) > 2:
                            count = 1
                            centre_1 = None
                            centre_2 = None
                            centre_3 = None
                            if a > 4:
                                a = 0
                                folder_num += 1
                            else:
                                a = 0

                    if a > 4 and hand_detector_flag == False:
                        if folder_num in time_stamps:
                            pass
                        else:
                            time_stamps[folder_num] = timestamp
                else:
                    a = 0
                    count = 1
                    centre_1 = None
                    centre_2 = None
                    centre_3 = None
        else:
            a = 0
            count = 1
            centre_1 = None
            centre_2 = None
            centre_3 = None
              
    cv2.imshow("video", frame)
    cv2.imshow("img_output", img_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("--------------------Time stamps of Stationary Objects--------------------")
print(time_stamps)
cap.release()
cv2.destroyAllWindows()

