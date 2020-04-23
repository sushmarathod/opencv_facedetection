# OpenCV Python program to detect cars in video frame 
# import libraries of python OpenCV 
import cv2 as cv

# capture frames from a video 
cap = cv.VideoCapture('video.avi') 

# Trained XML classifiers describes some features of some object we want to detect 
car_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml') 

# loop runs if capturing has been initialized. 
while True: 
    # reads frames from a video 
    ret, frames = cap.read() 

    if not ret: #if vid finish repeat
        frame = cv.VideoCapture("video.avi")
        continue
    if ret:  # if there is a frame continue with code
        # convert to gray scale of each frames 
        gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY) 

        # Detects cars of different sizes in the input image 
        cars = car_cascade.detectMultiScale(gray, 1.1, 1) 

        # To draw a rectangle in each cars 
        for (x,y,w,h) in cars: 
            cv.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
            
        # Display frames in a window 
        cv.imshow('video2', frames) 

        # Wait for Esc key to stop 
        if cv.waitKey(33) == 27: 
            break

# De-allocate any associated memory usage 
cv.destroyAllWindows() 
