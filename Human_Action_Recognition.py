import cv2
import time
import math as m
import mediapipe as mp


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree
# function to join landmarks
def line_draw():
    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
    #cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
    #cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

    # Join landmarks.
    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)

    cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
    cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

    cv2.line(image, (l_knee_x, l_knee_y), (l_hip_x, l_hip_y), green, 4)
    cv2.line(image, (l_knee_x, l_knee_y), (l_knee_x, l_knee_y - 100), green, 4)


    cv2.line(image, (l_ankle_x, l_ankle_y), (l_knee_x, l_knee_y), green, 4)
    cv2.line(image, (l_ankle_x, l_ankle_y), (l_ankle_x, l_ankle_y - 100), green, 4)

    cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), green, 4)
    cv2.line(image, (l_wrist_x, l_wrist_y), (l_wrist_x, l_wrist_y- 100), green, 4)

    cv2.line(image, (l_elbow_x, l_elbow_y), (l_shldr_x, l_shldr_y), green, 4)
    cv2.line(image, (l_elbow_x, l_elbow_y), (l_elbow_x, l_elbow_y - 100), green, 4)

# =============================CONSTANTS and INITIALIZATIONS===================================== #
# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


if __name__ == "__main__":
    # For webcam input replace file name with 0 in cv2.VideoCapture(0)
    #file_name = "D:/Downloads/batting.mp4"
    #file_name = "D:/Downloads/standing.mp4"
    #file_name = "D:/Downloads/sitting.mp4"
    #file_name="D:/Downloads/Tenis_service.mp4"
    #file_name="D:/Downloads/Chair_sitting.mp4"
    #file_name="D:/Downloads/ready_to_bat.mp4"
    #file_name="D:/Downloads/defence.mp4"
    #file_name="D:/Downloads/bouncer.mp4"
    #file_name="D:/Downloads/Squad.mp4"
    file_name="D:/Downloads/VID_20240418_083655009.mp4" # video with all actions(sitting,standing,ready to bat, straight drive, upper-cut shot)
    
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print("Frame size : ",frame_size)
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')*'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Video writer.
    video_output = cv2.VideoWriter('D:/Downloads/output_3.avi', fourcc, fps,frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        
        ########################## Acquire the landmark coordinates.  ################################
        # Once aligned properly, left or right should not be a concern.      
        # Left shoulder.
        if lm is not None:
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            # Left knee
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            # Left Ankle
            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
            # left foot index
            l_ft_idx_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)
            l_ft_idx_y = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)
            # Left Elbow
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            # left wrist
            l_wrist_x= int(lm.landmark[lmPose.LEFT_WRIST].x * w)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
            ########################################################################################################

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            #if offset < 100:
            #    cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
            #else:
            #    cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

            #######################  Calculate angles   ##########################################################
            #neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            #torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            leg_inclination=findAngle(l_ankle_x, l_ankle_y, l_knee_x, l_knee_y)
            thigh_inclination=findAngle(l_knee_x, l_knee_y, l_hip_x, l_hip_y)
            back_inclination=findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            lower_hand_inclination=findAngle(l_wrist_x, l_wrist_y, l_elbow_x, l_elbow_y)
            upper_hand_inclination=findAngle(l_elbow_x, l_elbow_y, l_shldr_x, l_shldr_y)
            ####################################################################################################
            #print(f'leg_inclination:{leg_inclination},thigh_inclination:{thigh_inclination},back_inclination:{back_inclination},lower_hand_inclination:{lower_hand_inclination},upper_hand_inclination:{upper_hand_inclination}')

            ################################ Draw landmarks.###############################################
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            #cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (l_ankle_x, l_ankle_y), 7, yellow, -1)
            cv2.circle(image, (l_knee_x, l_knee_y), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.circle(image, (l_wrist_x, l_wrist_y), 7, yellow, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            #cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            #cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            #cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

            if (10 < leg_inclination < 40) and (50<thigh_inclination<90) and (0<back_inclination<20):
                angle_text_string="Sitting"
                #print(angle_text_string)
                line_draw()

            elif (0 < leg_inclination < 20) and (0 <thigh_inclination< 10) and (0 <back_inclination<20):
                angle_text_string="Standing"
                #print(angle_text_string)
                line_draw()
            elif (0 < leg_inclination < 20) and (15 <thigh_inclination< 35) and (15 <back_inclination<35) and (0< lower_hand_inclination<35) and (10<upper_hand_inclination<30):
                angle_text_string="Ready to Bat"
                #print(angle_text_string)
                line_draw()
            elif (0 < leg_inclination < 17) and (15 <thigh_inclination< 30) and (20 <back_inclination<35) and (20< lower_hand_inclination<60) and (15<upper_hand_inclination<45):
                angle_text_string="Straight Drive"
                #print(angle_text_string)
                line_draw()  
            elif (5 < leg_inclination < 20) and (5 <thigh_inclination< 20) and (0 <back_inclination<10) and (100< lower_hand_inclination<155) and (120<upper_hand_inclination<170):
                angle_text_string="Upper-Cut Shot"
                #print(angle_text_string)
                line_draw()

            image=cv2.resize(image,frame_size)
                             
            # Write frames.
            video_output.write(image)

            # Display.
            cv2.imshow('Action Recognition', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

cap.release()
video_output.release()
cv2.destroyAllWindows()
