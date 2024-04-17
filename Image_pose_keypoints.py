import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Pose model
pose = mp_pose.Pose()

img_path = "D:/Downloads/milin.png"
image = cv2.imread(img_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Make inference
results = pose.process(image_rgb)

# Draw pose landmarks on the image
annotated_image = image.copy()
mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display annotated image
cv2.imshow("MediaPipe Pose Output", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
