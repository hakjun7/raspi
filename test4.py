import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('soocard2.jpg',0) # 라즈베리 파이로 찍은 명찰사진

imgs = {'kyungseok':'seokprofile.jpg',
        'hakjun':'hakprofile.jpg',
        'kyungsoo':'sooprofile.jpg'
        } #DB에 있는 학생들 사진들

match = 0
student = ''
for i in imgs.keys():
    img2 = cv2.imread(imgs[i],0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
    print(len(good))
    if match < len(good):
        match = len(good)
        student = i

print(f'반갑습니다 {student} 님')
img = cv2.imread(imgs[student],0)
# plt.imshow(img)


# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# plt.imshow(img3),plt.show()