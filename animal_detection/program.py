import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# link from: https://blogs.oracle.com/meena/cat-face-detection-using-opencv
# Download harcasscade xml files from :
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalcatface.xml
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalcatface_extended.xml

cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cat_ext_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

# set tunable parameters
SF = 1.05 # scale factor, can try different values of it like 1.3 etc
N = 3 # Minimum neighbours like 3,4,5,6 etc

def processImage(image_dir,image_filename):
    # read the image
    img = cv2.imread(image_dir+'/'+image_filename)
    # convery to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # this function returns tuple rectangle starting coordinates x,y, width, height
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    #print(cats) # one sample value is [[268 147 234 234]]
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    #print(cats_ext)
    
    # draw a blue rectangle on the image
    for (x,y,w,h) in cats:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)       
    # draw a green rectangle on the image 
    for (x,y,w,h) in cats_ext:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    # save the image to a file
    cv2.imwrite('out'+image_filename,img)

# for idx in range(1,7):
#     processImage('cats/',str(idx)+'.jpeg')

# import os
# all_cat_images = os.listdir('/media/sushma/9A2E19612E1937A91/workspace/Face_Detection/animal_detection/cats')
# print(all_cat_images)
# for i in all_cat_images:
#     processImage('cats/',i)

processImage('.','image7.jpg')

 