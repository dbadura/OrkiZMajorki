import cv2
from ocr_test import get_UIC_from_photo
import os
try:
    from PIL import Image
except ImportError:
    import Image

path = "/home/joanna/Dokumenty/Projekty/Prywatne/skyhacks/0_0_left_56.jpg"

image = cv2.imread(path)

def get_text_areas(image):
    """
    Get distinctive areas from image.
    :param image: cv2 image
    :return: list of images with selected areas
    """
    # Prepare images
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask) # Black image with white text

    # To manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 1))
    # dilate , more the iteration more the dilation
    dilated = cv2.dilate(image_final, kernel, iterations=9)

    # Find contours
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    # Get dimmensions for new area
    x1 = None
    x2 = None
    y1 = None
    y2 = None

    best_contours = []
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't choose small or to large false positives that aren't text
        if w < 100 or h < 35 or h > 100:
            continue
        # draw rectangle around contour on original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

        margin = 20
        y1 = (y-margin) if (y-margin) >= 0 else 0
        y2 = (y+h+margin) if (y+h+margin) <= image.shape[0] else image.shape[0]
        x1 = (x-margin) if (x-margin) >= 0 else 0
        x2 = (x+w+margin) if (x+w+margin) <= image.shape[1] else image.shape[1]
        cropped_text_area = image[y1:y2, x1:x2]
        best_contours.append(cropped_text_area)

    # cv2.imshow("areas", image)
    # cv2.waitKey()
    # write original image with added contours
    return(best_contours)


def get_uic_from_full_image(image):
    cropped_images = get_text_areas(image)
    detected_uic = []
    for img in cropped_images:
        detected_uic.append(get_UIC_from_photo(img))
    pass
    detected_uic = list(filter(None, detected_uic))
    # length = len(detected_uic)
    # if (length < 1):
    #     return(None)
    return(detected_uic)

file_path="./data/Training/Training/0_2/0_2_right/"
for filename in sorted(os.listdir(file_path)):
    if filename.endswith(".jpg") :
        image = cv2.imread(file_path+filename)
        UIC= get_uic_from_full_image(image)
        # if len(UIC) > 1:
        #     print(filename + ": " + UIC)
        # else:
        #     print(filename + "failed")
        # continue
        print(filename)
        print(UIC)
    else:
        continue
