import cv2
import numpy as np
import os
import pymysql
import board
import busio as io
import adafruit_mlx90614
import time
import sys

conn = pymysql.connect(host="localhost",
        user="b308",
        passwd="b308",
        db="ssafygate")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
student = [0,0]
cnt = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'hakjun', 'kyungsoo','moonseok','seunghyuk','dongwook','kyungseok'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
print('start recognize')
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            #id = names[id]
            confidence =(round(100 - confidence))
            cnt += 1
            if confidence > student[1]:
                student[0] = id
                student[1] = confidence
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    if cnt >= 5:
        print(f'hi {names[id]}')
        break
    # bring survey data
    #cv2.imshow('camera',img)
while True:
    k = input('sd:') # Press 'ESC' for exiting video
    if k == 'submib':
        try:
            with conn.cursor() as cur :
                sql="insert into survey (student_id, body_temparature, danger, body_check) values({},36.5,0,0)".format(id)
                cur.execute(sql)
                conn.commit()
        finally:
            conn.close()
        break
        
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
