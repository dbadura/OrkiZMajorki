#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import pytesseract
import os
import cv2 as cv
import numpy as np
import re
from IPython.display import display

filename = "0_42_left_57.jpg"
# Simple image to string

pattern = re.compile("(\d{7}\-\d{1})")

def get_UIC_from_photo(folder, filename):
#    ocr2 = pytesseract.image_to_string(Image.open(filename))
    img = cv.imread(folder + filename,0)
    img = cv.medianBlur(img,3)
    prog = np.average(img)*1.3
    ret,img = cv.threshold(img,prog,255,cv.THRESH_BINARY_INV)
#    img = cv2.resize(img, None, fx=5, fy=5)
#    ocr1 = pytesseract.image_to_string(filename)    # Convert to gray
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#    kernel = np.ones((4, 4), np.uint8)
#    img = cv2.erode(img, kernel, iterations=2)
    ocr1 = pytesseract.image_to_string(img)#, config='--psm 3')
#    cv2.imwrite('test1.jpg', img)
#    print(ocr1)
#    if ocr1 != ocr2:
#        print(ocr1)
#        print(ocr2)
    ocr1 = re.sub('~', '-',ocr1)
    ocr1 = re.sub('[A-Za-z (\n)]', '',ocr1)
    
    ocr1 = ocr1.replace(" ", "")
    m = pattern.search(ocr1)
    if m:
        return m[1]
#        print("matching ", m[1])
    if len(ocr1)>7:
        return '999'
    return ""
        

#print(get_UIC_from_photo('test1.jpg'))
UIC_list = list()
folder = os.getcwd() + '/'
folder = "/media/dawid/Dane 2_2/Projekty/Python/Hackathon/Training/Training/0_4/0_4_left/"
for filename in os.listdir(folder):
    if filename.endswith(".jpg") :
        UIC= get_UIC_from_photo(folder, filename)
        if len(UIC) > 1:
            print(filename + ": " + UIC)
        else:            
            print(filename + " failed")
            
        if len(UIC) > 8:
            UIC_list.append(filename + ": " + UIC)
        continue
    else:
        continue