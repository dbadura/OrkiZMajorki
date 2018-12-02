#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import pytesseract
import os
import cv2 as cv
import numpy as np
import re

# UIC pattern regex
#pattern = re.compile("(\d{7}\-\d{1})") # coreect pattern
pattern = re.compile("(\d{6,7}\-\d{1})") # pattern for one number shorter than correct UIC

def get_UIC_from_photo(image):
    imgo = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert RGB image to grayscale
    img = cv.medianBlur(imgo,3) # median blur filter for removing noise 
    
    # adaptive thresholding for convert image from grayscale to binary
    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,49,-15)
    
    # run OCR on image
    ocr1 = pytesseract.image_to_string(img)#, config='--psm 3')
    
    # replace some character to other ones which are similar
    ocr1 = re.sub('~', '-',ocr1)
    ocr1 = re.sub('\$', '5',ocr1)
    
    # remove all characters which are not numbers or "-" sign
    ocr1 = re.sub('([^0-9\-]*)', '',ocr1)
    
    # search for UIC pattern
    m = pattern.search(ocr1)
    
    if m:
        if len(m[1]) == 8: # if this code is one number to shotr
            return '5' + str(m[1]) # add 5 on the begining
        else:
            return m[1] # return matched UIC
    else:       
        img = cv.medianBlur(imgo,5) # median blur filter for removing noise 
        
        # adaptive thresholding for convert image from grayscale to binary
        img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,49,-15)
        
        # run erosion operation
        kernel = np.ones((2,2),np.uint8)
        img = cv.erode(img,kernel,iterations = 3)
        
        # run OCR on image
        ocr1 = pytesseract.image_to_string(img)#, config='--osm 9')
        
        # replace some character to other ones which are similar
        ocr1 = re.sub('~', '-',ocr1)
        ocr1 = re.sub('\$', '5',ocr1)
        
        # remove all characters which are not numbers or "-" sign
        ocr1 = re.sub('([^0-9\-]*)', '',ocr1)
        
        # search for UIC pattern
        m = pattern.search(ocr1)
        
        if m:
            if len(m[1]) == 8: # if this code is one number to shotr
                return '5' + str(m[1]) # add 5 on the begining
            else:
                return m[1] # return matched UIC
    return ""
