import os
os.chdir("E:\College\Final year project\images")
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
l=["15-LK-10898" , "TN 52 M6912" , "MH12DE1433" , "TN-55 AR 2666" , "TN55 AR 2666"]

import numpy as np
import cv2
import matplotlib.pyplot as plt

import cv2
cap = cv2.VideoCapture(0)
add="https:192.168.168.15:8080/video"
cap.open(add)
while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame123',gray)
    c_edge = cv2.Canny(gray, 170, 200)
    contours,h = cv2.findContours(c_edge,1,2)
    largest_rectangle = [0,0]
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4: 
            area = cv2.contourArea(cnt)
            if area > largest_rectangle[0]:
                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
    
    x,y,w,h = cv2.boundingRect(largest_rectangle[1])
    roi=frame[y:y+h,x:x+w]
    cv2.drawContours(frame,[largest_rectangle[1]],0,(0,0,255),-1)
    plt.imshow(roi, cmap = 'gray')
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    text = pytesseract.image_to_string(roi)
    
    if(text):
        print(text)
    if cv2.waitKey(1) & 0xFF == ord('f'):
        text = pytesseract.image_to_string(roi)
        break
cap.release()
cv2.destroyAllWindows()        
