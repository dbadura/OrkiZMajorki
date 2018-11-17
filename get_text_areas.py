import cv2
try:
    from PIL import Image
except ImportError:
    import Image

path = "/home/joanna/Dokumenty/Projekty/Prywatne/skyhacks/0_56_left_26.jpg"

image = cv2.imread(path)

def get_text_areas(image):

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

    best_contours = []
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't choose small or to large false positives that aren't text
        if w < 100 or h < 35 or h > 100:
            continue
        # # draw rectangle around contour on original image
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cropped_text_area = image[y:y+h, x:x+w]
        best_contours.append(cropped_text_area)
    # write original image with added contours
    return(best_contours)

cropped_images = get_text_areas(image)
