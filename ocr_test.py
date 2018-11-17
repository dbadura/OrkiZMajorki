#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytesseract
import cv2 as cv
import numpy as np
import re

pattern = re.compile("(\d{7}\-\d{1})")

def get_UIC_from_photo(img):

    img = cv.medianBlur(img,3)
    prog = np.average(img)*1.3
    ret,img = cv.threshold(img,prog,255,cv.THRESH_BINARY_INV)
    ocr1 = pytesseract.image_to_string(img)

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
