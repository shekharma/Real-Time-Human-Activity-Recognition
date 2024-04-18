# Real-Time-Human-Activity-Recognition
Human activity recognition is combination pose estimation using keypoints and landmarks and classify them into their categories like sitting, standing or laying etc. 
### Objective: 
Classification of human action form recorded video as well as in real-time

### Library :
  1.  OpenCV
  2.  Mediapipe
### Implementation:
  1. Keypoints and Pose Landmark finding : After importing required library and reading our video file through the frames. We use inbuilt functions of mediapipe library to extract the required keypoints of our body. So here I'm constraining my model by allowing it to use only a left side veiw to detect a action.
  2. Inclination Angle Calculation: Just extracting the keypoints won't help to recognize the action. Extracted keypoints are use to calculate the inner angle.
![IMG_20240417_212249720~3](https://github.com/shekharma/Real-Time-Human-Activity-Recognition/assets/122733304/8bb5802b-5b23-43ce-ae97-c908ff821a4b)
 3. By using same approach I calculated ankle-knee, knee-hip, hip-shoulder, wrist-elbow and elbow-shoulder inclination angles to recognize the activity.

By analyzing the distribution of angles for a particular activities, I passed them through a coditional loop to recognize the activity.
You can see the results with activity name in upper left corner in the activity_output.mp4 files. 


### Reference
  -  https://github.com/google/mediapipe
