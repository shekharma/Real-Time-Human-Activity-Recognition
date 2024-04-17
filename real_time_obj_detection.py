import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import os
import random
#from moviepy.editor import VideoFileClip
import math
import sys
import random
import csv
from sys import argv
from imageai.Detection import ObjectDetection
from time import time
#from imageai.Detection.keras_retinanet.utils.colors import label_color  
from scipy.spatial import distance
import numpy as np
import pandas as pd
from time import time
from time import sleep
#import vlc//
import socket
import threading
from threading import Lock
x=random.random()
model_path="D:/Downloads/yolov3.pt"
#model_path ="D:/Downloads/yolo.h5"
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
#custom object detection
custom = detector.CustomObjects(person=True)
##custom = detector.CustomObjects(person=True)
# Set up Mediapipe model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose=mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


#########
## This is code to connect real time camera to detect objetc
#UDP_IP = "127.0.0.1"
##UDP_IP = "192.168.0.23"
#UDP_PORT = 1999
##MESSAGE = b"Hello, World!"
#MESSAGE=b""
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
#sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
########
def cal_distance(mid_v,num):
    dist = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1,num):
            if i!=j:
                dst = distance.euclidean(mid_v[i], mid_v[j])
                dist[i][j]=dst
    return dist

def find_closest(dist,num,thresh):
    p1=[]
    p2=[]
    d=[]
    for i in range(num):
        for j in range(i,num):
            if( (i!=j) & (dist[i][j]<=thresh)):
                p1.append(i)
                p2.append(j)
                d.append(dist[i][j])
    return p1,p2,d


cap = cv.VideoCapture(0)#(0=laptop camera,1=another attached camera)    
#video_path = "D:/Downloads/VID_20240415_211655480.mp4"
#video_path="D:/Downloads/VID_20240416_193400127.mp4"
#cap = cv.VideoCapture(video_path)
#fps = int(cap.get(cv.CAP_PROP_FPS)) # get fps of input video
#print(f'frame_per_sec{fps}')
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) # get size of input video

# save_path = r'input_save_2.mp4'
out = cv.VideoWriter(f'D:/Downloads/output_{x}.mp4',cv.VideoWriter_fourcc(*'mp4v'), 10, (size)) #create a video format to save processed frames into it

#setting camera resolution
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

latest_frame = None
last_ret = None
lo = Lock()

def rtsp_cam_buffer(vcap):
    global latest_frame, lo, last_ret
    while True:
        with lo:
            last_ret, latest_frame = vcap.read()

t1 = threading.Thread(target=rtsp_cam_buffer,args=(cap,),name="rtsp_read_thread")
t1.daemon=True
t1.start()
print("Thread started")

frame_skipper = 0

#frame_counter = 0
while(cap.isOpened()):
    if (last_ret is not None) and (latest_frame is not None):
        img = latest_frame.copy()

    else:
        print("unable to read the frame")
        #sleep(0.2)

    sending_string = ""

    if last_ret:
        ret, frame = cap.read()
        if(ret == False):   # reached the end of video
            break
        frameH = frame.shape[0] #830
        frameW = frame.shape[1] #1440
        displayed_frameH = 1080
        displayed_frameW = 1920
        ratio_h = displayed_frameH / frameH
        ratio_w = displayed_frameW / frameW
        person=[]
        mid_v=[]
        bb_box=[]
##        print("width:", frameW, "Height:", frameH)
   
        our_time = time()
        #mp_object_detection = mp.solutions.object_detection
        #object_detection = mp_object_detection.ObjectDetection()
        #results = object_detection.process(frame)
        #detections = results.detections
        #returned_image = frame.copy()
        returned_image, detections = detector.detectObjectsFromImage(custom_objects=custom, input_image=frame, output_type="array", minimum_percentage_probability=30)
        # Check if returned_image is None
        

    # Resize the image
        #resized_image = cv.resize(returned_image, (displayed_frameW, displayed_frameH))
        resized_image = cv.resize(returned_image, (600,800))

    # Get the height and width of the resized image
        height = resized_image.shape[0]
        width = resized_image.shape[1]

    # Further processing...

        
        
        resize = cv.resize(returned_image, (displayed_frameW, displayed_frameH))#, interpolation = cv.INTER_LINEAR)
        heigh = resize.shape[0]
        widt = resize.shape[1]
        #print("width:", heigh, "Height:", widt)
#        MESSAGE = bytes((str(len(detections))),'ascii')
        MESSAGE_1 = str(len(detections))
    #cv.waitKey(10000)
        sending_string = "Detected, " + MESSAGE_1 + ", "
        no_of_p = 0
        for detection in detections:
            person.append(detection["name"])
    #        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
            x1=detection["box_points"][0]
            y1=detection["box_points"][1]
            x2=detection["box_points"][2]
            y2=detection["box_points"][3]

            # calculate ROI coordinates and size
            roi_x = x1
            roi_y = y1
            roi_width = x2 - x1
            roi_height = y2 - y1

            x_mid = int(int((x1+x2)/2) * ratio_w)
            y_mid = int(int(y2) * ratio_h)
            if(no_of_p == len(detections)-1):
                sending_string = sending_string + "p" + str(no_of_p) + ", " + str(x_mid) + ", " + str(y_mid)
            else:
                sending_string = sending_string + "p" + str(no_of_p) + ", " + str(x_mid) + ", " + str(y_mid) + ", "

            mid   = (x_mid,y_mid)
            mid_v.append(mid)
            bb_box.append(detection["box_points"])
            #print (mid_v)
            no_of_p = no_of_p+1
            #print("IT TOOK : ", time() - our_time)
       
            num=len(mid_v)
            dist=cal_distance(mid_v,num)
            thresh = 250
            p1,p2,d=find_closest(dist,num,thresh)
            risky = np.unique(p1+p2)
           
            counter = 0    
    #         for i in range(len(mid_v)):
    #            counter += 1
    #            _ = cv.circle(resize, mid_v[i], 5, (0, 255, 255), -1)
    #            cv.putText(resize, str(counter), mid_v[i], cv.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv.LINE_AA)
           
    #            if (any(m==i for m in risky) == True):
    #                cv.rectangle(resize, (int(bb_box[i][0]*ratio_w), int(bb_box[i][1]*ratio_h)), (int(bb_box[i][2]*ratio_w),int(bb_box[i][3]*ratio_h)), (0, 0, 255), 3)
    #            else:
    #                cv.rectangle(resize, (int(bb_box[i][0]*ratio_w), int(bb_box[i][1]*ratio_h)), (int(bb_box[i][2]*ratio_w),int(bb_box[i][3]*ratio_h)), (0, 255, 0), 3)
    # # roi = returned_image[bb_box[1]:bb_box[1]+bb_box[3],bb_box[0]:bb_box[0]+bb_box[2]]    
        # if roi is not None:
        #             blurred_roi = cv.medianBlur(roi,31)
        #             blurred_roi = cv.GaussianBlur(blurred_roi,(51,51),75)
        #             returned_image[bb_box[1]:bb_box[1]+bb_box[3],bb_box[0]:bb_box[0]+bb_box[2]] = blurred_roi
            # image = cv.rectangle(resize, (int(bb_box[i][0]*ratio_w), int(bb_box[i][1]*ratio_h)), (int(bb_box[i][2]*ratio_w),int(bb_box[i][3]*ratio_h)))    
            # roi = frame[int(bb_box[i][1]*ratio_h):int(bb_box[i][3]*ratio_h),int(bb_box[i][0]*ratio_w):int(bb_box[i][2]*ratio_w)]
            # if roi is not None:
            #     blurred_roi = cv.medianBlur(roi,31)
            #     blurred_roi = cv.GaussianBlur(blurred_roi,(51,51),75)
            #     frame[int(bb_box[i][1]*ratio_h):int(bb_box[i][3]*ratio_h),int(bb_box[i][0]*ratio_w):int(bb_box[i][2]*ratio_w)] = blurred_roi
       
        # extract the ROI image
            roi_image = returned_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            

            # Use Mediapipe to extract pose features
            image = cv.cvtColor(roi_image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                height, width, channels = roi_image.shape
                for landmark in landmarks:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    # cv.circle(roi_image, (x, y), 5, (255, 0, 0), -1)
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(roi_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    returned_image[y1:y2, x1:x2] = roi_image
            #     annotated_image = roi_image.copy()
            #     mp_drawing.draw_landmarks(
            #     annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # # Extract the required pose features
            #     landmarks = results.pose_landmarks.landmark
            #     # Do something with the pose features
               
        # save the ROI image
               

            # cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(returned_image)

            cv.imshow("image_roi", returned_image)


    if cv.waitKey(1) & 0xFF == ord('q'):  ## press q key to stop
        break  


cap.release()
out.release()
cv.destroyAllWindows()
