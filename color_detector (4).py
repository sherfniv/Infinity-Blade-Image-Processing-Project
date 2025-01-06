from ctypes.wintypes import RGB
import cv2
import numpy as np

class ColorDetector:
    def __init__(self,color):
        # Create mask for red color
        if color == 1:
            #self.low_color = np.array([70, 106, 31])
            #self.high_color = np.array([100, 247, 167])
            self.low_color = np.array([47, 91, 46])
            self.high_color = np.array([95, 255, 134])
        if color == 2:
            self.low_color = np.array([113, 50, 136])
            self.high_color = np.array([127, 255, 255])
        
    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks with color ranges
        mask = cv2.inRange(hsv_img, self.low_color, self.high_color)
        fgmask = fgbg.apply(mask)
        
        # Find Contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        max_contour_area = 0
        max_contour = None
        area_thresh = 40  # Ignore contours with a small area
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_contour_area and area > area_thresh:
                max_contour_area = area
                max_contour = cnt
            #    print(f"area: {area}")
            break
        # Bitwise-AND mask and original image
        #res = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.imshow('image', res)
        return max_contour
    
    def undetect(self, frame):

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create masks with color ranges
        mask = cv2.bitwise_not(cv2.inRange(hsv_img, self.low_color, self.high_color))
        return mask
    
cap = cv2.VideoCapture(0)
cd = ColorDetector()
fgbg = cv2.createBackgroundSubtractorMOG2()

cv2.destroyAllWindows()