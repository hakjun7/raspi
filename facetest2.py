import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')

imgs = {'kyungseok':'seokprofile.jpg',
        'hakjun':'hakprofile.jpg',
        'kyungsoo':'sooprofile.jpg'
        }
match = 0
student = ''

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]    
    
    if len(faces) == 1:
        for i in imgs.keys():
            img2 = cv2.imread(imgs[i],0)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.4*n.distance:
                    good.append([m])
            print(len(good))
            if match < len(good):
                match = len(good)
                student = i

        print(f'반갑습니다 {student} 님')        
        
    
    if ret:
        #cv2.imshow('camera-o',img)
        if cv2.waitKey(1) & 0xFF == 27:
            print(match)
            break
    else:
        print('no camera')
        break
cap.release()
cv2.destroyAllWindows()