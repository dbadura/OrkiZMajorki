#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:11:22 2018

@author: dawid

this code is not used in final solution
just for testing different algoithms
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import pytesseract


imgc = cv.imread('0_0_left_26.jpg')
#imgc[:,:,2] = 0
img = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(img,5)
prog2 = np.average(img)*1.2

prog = prog2
#kernel = np.ones((2, 1), np.uint8)

ret,thresh1 = cv.threshold(img,prog,255,cv.THRESH_BINARY)
ret,img = cv.threshold(img,prog,255,cv.THRESH_BINARY_INV)
#ret,thresh3 = cv.threshold(img,prog,255,cv.THRESH_TRUNC)
#ret,thresh4 = cv.threshold(img,prog,255,cv.THRESH_TOZERO)
#ret,thresh5 = cv.threshold(img,prog,255,cv.THRESH_TOZERO_INV)
#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#images = [img, thresh1, thresh2]#, thresh3, thresh4, thresh5]
#for i in range(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
plt.subplot(1,1,1),plt.imshow(img,'gray')
#plt.subplot(1,1,1),plt.imshow(img)
plt.show()
#cv.imwrite('test1.jpg', images[2])



img = img
ocr1 = pytesseract.image_to_string(img)#, config='--psm 3')
print(ocr1)
