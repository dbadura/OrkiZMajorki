#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import pytesseract
import os
import cv2 as cv
import numpy as np
import re

filename = "0_42_left_57.jpg"
# Simple image to string

#pattern = re.compile("(\d{7}\-\d{1})")
pattern = re.compile("(\d{6,7}\-\d{1})")

def get_UIC_from_photo(image):
    imgo = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(imgo,3)
#    prog = np.average(img)+np.mean(img)/6
#    prog = 167
    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,49,-15)
    ocr1 = pytesseract.image_to_string(img)#, config='--psm 3')
    ocr1 = re.sub('~', '-',ocr1)
    ocr1 = re.sub('\$', '5',ocr1)
    ocr1 = re.sub('([^0-9\-]*)', '',ocr1)
    
#    plt.subplot(1,1,1),plt.imshow(img,'gray')
#    #plt.subplot(1,1,1),plt.imshow(img)
#    plt.show()
    m = pattern.search(ocr1)
    
    if m:
        if len(m[1]) == 8:
            return '5' + str(m[1])
        else:
            return m[1]
    else:       
        img = cv.medianBlur(imgo,5)
        img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,49,-15)
        kernel = np.ones((2,2),np.uint8)
        img = cv.erode(img,kernel,iterations = 3)
        
        ocr1 = pytesseract.image_to_string(img)#, config='--osm 9')
        
        ocr1 = re.sub('~', '-',ocr1)
        ocr1 = re.sub('\$', '5',ocr1)
        ocr1 = re.sub('([^0-9\-]*)', '',ocr1)
        
        m = pattern.search(ocr1)
        
        if m:
            if len(m[1]) == 8:
                return '5' + str(m[1])
            else:
                return m[1]
#    if len(ocr1)>7 and len(ocr1)<10:
#        return '999'
    return ""
