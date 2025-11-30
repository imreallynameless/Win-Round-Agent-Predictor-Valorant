# Use this to get coordinates from a screenshot to help determine ROI boundaries and dimensions

import cv2

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked:", x, y)

img = cv2.imread("screenshots/val.png")
cv2.imshow("image", img)
cv2.setMouseCallback("image", click)
cv2.waitKey(0)