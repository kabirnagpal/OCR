from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


"""
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    print(frame)
else:
    ret = False
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()
"""


imgpath="images/img11.jpg"
img=cv2.imread(imgpath,0)


# load the image and compute the ratio of the old height to the new height, clone it, and resize it
ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height = 500)


img = cv2.GaussianBlur(img, (5, 5), 0)
edged = cv2.Canny(img, 75, 200)


cv2.imshow("Edged", edged)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        screenCnt = approx
        break
        
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)

# apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255


# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("final/image1.jpg",warped)