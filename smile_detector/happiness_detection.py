# Face detection

import cv2

# load the cascades these are trained classifers
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'./haarcascade_smile.xml')


# function to detect
# gray is the gray scale image
# frame is the original image
def detect(gray, frame):
    # get dimensions and coordiates of the detection
    # at least 5 neighbor zones should overlap for detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # draw the rect on where the face is
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (e_x, e_y, e_w, e_h) in eyes:
             # draw the rect on where the eyes are
             cv2.rectangle(roi_color, (e_x, e_y), (e_x+e_w, e_y+e_h), (0, 255, 0), 2)
        # smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 23)
        for(sx, sy, sw, sh) in smiles:
            #draw smile rect
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)
    return frame

# Applying face detection to webcam video
video_capture = cv2.VideoCapture(0) #contains the last frame on the webcam
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    #display the detected images
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# turn off webcam
video_capture.release()
# destory the widows
cv2.destroyAllWindows()