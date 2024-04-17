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



# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

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
    # For webcam input replace file name with 0.
    #file_name = "D:/Downloads/batting.mp4"
    #file_name = "D:/Downloads/standing.mp4"
    #file_name = "D:/Downloads/sitting.mp4"
    file_name="D:/Downloads/VID_20240417_175945099.mp4"
    #cap = cv2.VideoCapture(0)  # To access the camera -real time action recognition
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('D:/Downloads/output_1.mp4', fourcc, fps, frame_size)

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
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        #angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        #if neck_inclination < 60 and torso_inclination < 30:
        if (10 < leg_inclination < 40) and (50<thigh_inclination<90) and (0<back_inclination<20):
            #bad_frames = 0
            #good_frames += 1
            angle_text_string="Sitting"
            print(angle_text_string)
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            #cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            #cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

            # Join landmarks.
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

        elif (0 < leg_inclination < 5) and (0 <thigh_inclination< 5) and (0 <back_inclination<5):
            #good_frames = 0
            #bad_frames += 1
            angle_text_string="Standing"
            print(angle_text_string)
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            #cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            #cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

            # Join landmarks.
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
        
        elif (0 < leg_inclination < 20) and (0 <thigh_inclination< 50) and (0 <back_inclination<35) and (50< lower_hand_inclination<75) and (0<upper_hand_inclination<15) :
            #good_frames = 0
            #bad_frames += 1
            angle_text_string="Batting Pose"
            print(angle_text_string)
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            #cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            #cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

            # Join landmarks.
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
            
    
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
