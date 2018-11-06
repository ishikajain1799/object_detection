
# coding: utf-8

# In[ ]:


import cv2 
import numpy as np 
def sift_detector(new_image,image_template):
    image1 = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    image2 = image_template
    
    sift = cv2.SIFT()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)
    
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    search_params = dict(checks = 100)
    
    flann = cv2.FlannBaseMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return len(good_matches)

cap = cv2.VideoCapture(0)

image_template = cv2.imread('C:\\Users\\Admin\\Desktop\\image1.jpg',0)

while True :
    ret, frame = cap.read()
    
    height, width = frame.shape[:2]
    
    top_left_x = width / 3
    top_left_y = (height/2) +(height/4)
    bottom_right_x = (width/3) *2
    bottom_right_y = (height/2) - (height/4)
    
    cv2.rectangle(frame,(top_left_x,top_left_y), (bottom_right_x, bottom_right_y), 255, 3)
    cropped = frame[bottom_right_y:top_left_y, top_left_x: bottom_right_x]
    
    frame = cv2.flip(frame,1)
    
    matches = sift_detector(cropped, image_template)
    
    cv2.putText(frame,str(matches),(450,450),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),1)
    
    threshold = 10
    
    if matches > threshold :
        cv2.rectangle(frame,(top_left_x,top_left_y), (bottom_right_x, bottom_right_y),(0,255,0),3)
        cv2.putText(frame, 'object found', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    
    cv2.imshow('object ddetection using sift', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

