import os
os.chdir("E:\College\Final year project\images")
def main():
    import numpy as np
    import cv2
    import imutils
    import sys
    import pytesseract
    import pandas as pd
    import time
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
    img = cv2.imread("1.jpg")
    img = imutils.resize(img, width=500)
    cv2.imshow("Original Image", img)  
    cv2.waitKey(0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Preprocess 1 - Grayscale Conversion", gray_img) 
    cv2.waitKey(0)
    gray_img = cv2.bilateralFilter(gray_img, 11, 17, 17)
    cv2.imshow("Preprocess 2 - Bilateral Filter", gray_img)   
    cv2.waitKey(0)
    c_edge = cv2.Canny(gray_img, 170, 200)
    cv2.imshow("Preprocess 3 - Canny Edges", c_edge)    
    cv2.waitKey(0)
    cnt, new = cv2.findContours(c_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCount = None
    im2 = img.copy()
    cv2.drawContours(im2, cnt, -1, (0,255,0), 3)
    cv2.imshow("Top 30 Contours", im2)      
    cv2.waitKey(0)
    count = 0
    for c in cnt:
        perimeter = cv2.arcLength(c, True)    
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:         
            NumberPlateCount = approx
            break
    masked = np.zeros(gray_img.shape,np.uint8)
    new_image = cv2.drawContours(masked,[NumberPlateCount],0,255,-1)
    new_image = cv2.bitwise_and(img,img,mask=masked)
    cv2.imshow("4 - Final_Image",new_image)  
    cv2.waitKey(0)
    configr = ('-l eng --oem 1 --psm 3')
    text_no = pytesseract.image_to_string(new_image, config=configr)
    data = {'Date': [time.asctime(time.localtime(time.time()))],
        'Vehicle_number': [text_no]}
    df = pd.DataFrame(data, columns = ['Date', 'Vehicle_number'])
    df.to_csv('Dataset_VehicleNo.csv')
    cv2.waitKey(0)
    flag=0
    import seaborn as sn 
    data=pd.read_csv(r"C:\Users\user\Documents\MATLAB\testdata.csv")
    reg=data['Registration Number']
    abc=text_no
    for idx,i in enumerate(reg):
        if(i==abc):
            #print(data.loc[idx,:])
            flag=1
    if(flag==0):
        print("Not registered")
    print(text_no)    
if __name__ == '__main__':
    main()