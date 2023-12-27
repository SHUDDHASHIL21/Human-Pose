import cv2
import mediapipe as mp
import numpy as np


pose_det = mp.solutions.pose
pose_draw = mp.solutions.drawing_utils
pose = pose_det.Pose()


#cap = cv2.VideoCapture("D:/Human Pose Tracking/train_data//Movement Demo - The Hang Clean.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (600, 400))
    

    
    results = pose.process(img)
    pose_draw.draw_landmarks(img, results.pose_landmarks, pose_det.POSE_CONNECTIONS ,
                             pose_draw.DrawingSpec((0 , 0 , 255) , 2 , 2) ,
                             pose_draw.DrawingSpec((0 , 255 , 0) , 2 , 2))


    h , w , c = img.shape
    opImg = np.zeros([h , w , c])
    #opImg.fill(255)
    pose_draw.draw_landmarks(opImg, results.pose_landmarks , pose_det.POSE_CONNECTIONS ,
                             pose_draw.DrawingSpec((0 , 0 , 255) , 2 , 2) ,
                             pose_draw.DrawingSpec((0 , 255 , 0) , 2 , 2))
    cv2.imshow("Extracted Pose" , opImg)    



    print(results.pose_landmarks)
    cv2.imshow("Pose Estimation", img)
    key = cv2.waitKey(1) & 0xFF  # Masking to get only the last 8 bits

    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
