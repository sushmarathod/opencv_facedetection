import cv2 as cv

original_image = cv.imread("puppy.jpg")
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
# Detect faces
detected_faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3)

face_found = False   #---Initially set the flag to be False
for (col, row, width, height) in detected_faces:
    if width > 0 :                 #--- Set the flag True if w>0 (i.e, if face is detected)
        face_found = True

        cv.rectangle(
            original_image,
            (col,row),
            (col+width,row+height),
            (0,255,0),
            2 # thickness
            )  #--- highlight the face   
        cv.putText(original_image,'FACE',(col+20,row+20), 2, 1, (0,255,0)) #---write the text
        small = cv.resize(original_image, (0,0), fx=0.3, fy=0.3)
        cv.imshow('Image', small)
    
cv.waitKey(0)
cv.destroyAllWindows()

