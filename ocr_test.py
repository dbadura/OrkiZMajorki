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

pattern = re.compile("(\d{7}\-\d{1})")


def get_UIC_from_photo(image):
    #    ocr2 = pytesseract.image_to_string(Image.open(filename))
    #     img = cv.imread(folder + filename,0)
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 3)

    prog = np.average(img) + np.mean(img) / 6
    ret, img = cv.threshold(img, prog, 255, cv.THRESH_BINARY_INV)
    ocr1 = pytesseract.image_to_string(img)  # , config='--psm 3')
    ocr1 = re.sub('~', '-', ocr1)
    ocr1 = re.sub('[A-Za-z (\n),]', '', ocr1)

    m = pattern.search(ocr1)
    if m:
        return m[1]
    if len(ocr1) > 7:
        return '999'
    return "0"
