import cv2 as cv,cv2
import numpy as np
from time import time as t_now
from pathlib import Path

class Project:

    def __init__(self):
        
        self.images = {}
        self.elements = []

        self.load_images()
    

    def load_images(self,filespath = "train"):
        images_dir = Path(filespath)
        images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
        for image_path in images_paths:
            img_color = cv.imread(str(image_path))
            if img_color is None:
                print(f'Error loading img_color {image_path}')
                continue
            self.images[image_path.name] = img_color
 
    @staticmethod
    def resize(img_color = None,scale_percent :int =30):
        width = int(img_color.shape[1] * scale_percent / 100)
        height = int(img_color.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv.resize(img_color, dim, interpolation=cv.INTER_AREA)
    
    def new_(self):
        images = [img for img in self.images.values()]
        
        window_name = "Test"               
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.images)-1, lambda x: x)
        cv.createTrackbar('HMin', window_name, 0, 255, lambda x: x)
        cv.createTrackbar('SMin', window_name, 0, 255, lambda x: x)
        cv.createTrackbar('VMin', window_name, 0, 255, lambda x: x)
        cv.createTrackbar('HMax', window_name, 0, 255, lambda x: x)
        cv.createTrackbar('SMax', window_name, 0, 255, lambda x: x)
        cv.createTrackbar('VMax', window_name, 0, 255, lambda x: x)

        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', window_name, 255)
        cv.setTrackbarPos('SMax', window_name, 255)
        cv.setTrackbarPos('VMax', window_name, 255)

        # Initialize HSV min/max values
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0
        image_num = 1


        while(1):
            img_color = images[image_num] 
           
            # Get current positions of all trackbars
            hMin = cv.getTrackbarPos('HMin', window_name)
            sMin = cv.getTrackbarPos('SMin', window_name)
            vMin = cv.getTrackbarPos('VMin', window_name)
            hMax = cv.getTrackbarPos('HMax', window_name)
            sMax = cv.getTrackbarPos('SMax', window_name)
            vMax = cv.getTrackbarPos('VMax', window_name)
            image_num = cv.getTrackbarPos('Image', window_name)
            img = img_color.copy()

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            gray = cv2.medianBlur(gray, 5)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            thresh = thresh.astype(np.uint8)
            color = cv2.bilateralFilter(img, 9, 250, 250)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255,0,0), thickness=cv2.FILLED)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            # white = cv.inRange(hsv, (27,0,172), (127,51,255))
            yellow_green_red_blue = cv.inRange(hsv, (0,90,0), (179,255,255))
            # colors = cv.inRange(hls, (0,57,0), (255,232,68))
            # green = cv.inRange(hls, (34,0,0), (131,93,255))
            # colors = 255 - colors
            # hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
            # mask = cv.inRange(hsv, lower, upper)
            # mask = 255 - mask
            # masks = colors + white + green
            # masks = colors + green + mask

            result = cv.bitwise_and(img, img, mask=yellow_green_red_blue)
           
            # cv2.drawContours(img_gray, contours, -1, [0, 255, 0], thickness=cv2.FILLED)
            cv2.imshow(window_name, self.resize(img,20))

            # Print if there is a change in HSV value
            if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display result img_color
            # cv.imshow(window_name, np.hstack([self.resize(img_color,20), self.resize(image,20)]))
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

cv.destroyAllWindows()

if __name__ == "__main__":
    start_time = t_now()
    print("Start time: ", start_time)
    win = Project()
    print("--- %s seconds ---" % (t_now() - start_time))

    win.new_()



