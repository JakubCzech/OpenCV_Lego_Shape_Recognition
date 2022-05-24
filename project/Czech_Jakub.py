from typing_extensions import Self
import cv2 as cv,cv2
from os import listdir
import numpy as np
import time
class lab:
        
    def __init__(self):
        self.img_color_list = []
        self.img_grey_list = []
        self.img_masked = []
        self.img_masked_gray = []
        self.load_images()
       
    def load_images(self):
        for file in listdir("SRC/train"):
            if file.endswith(".jpg"):
                img_color = cv.imread("SRC/train/" + file)
                img_grey = cv.imread("SRC/train/" + file, cv.IMREAD_GRAYSCALE)
                _, thresh1 = cv2.threshold(img_grey, 102, 255, cv2.THRESH_TOZERO)
                self.img_color_list.append(img_color)
                self.img_grey_list.append(thresh1)
                self.img_masked.append(self.get_color_shape(img_color))   
                self.img_masked_gray.append(cv.cvtColor(self.get_color_shape(img_color),cv.COLOR_BGRA2GRAY))
   
    @staticmethod
    def resize(image = None,scale_percent :int =20):
        """
        
        :param scale_percent:
        :param image:
        
        :return: resized image as np.ndarray
         """
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def get_color_shape(image):
        frame_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (0, 0, 0), (180, 107, 255))
        frame_threshold = cv.bitwise_not(frame_threshold)
        img = cv.bitwise_and(image, image, mask=frame_threshold)
 
        return img

    @staticmethod
    def get_edges(image):
        return cv.Canny(image,310,181)

    def show_masked(self):
        window_name = "Image after masked"
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.img_grey_list)-1, lambda x: x)
        image_num = 1
        while True:
            img_color = self.img_masked[image_num]
            img_ori = self.img_color_list[image_num]
            cv.imshow(window_name, self.resize(img_color))
            cv.imshow("Original", self.resize(img_ori))
            image_num = cv.getTrackbarPos('Image', window_name)
            key_code = cv.waitKey(10)
            if key_code == 27:
                break

        cv.destroyAllWindows()
        return


    def tuning(self):
        window_name = "tuning"
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.img_grey_list)-1, lambda x: x)
        cv.createTrackbar('Canny_1', window_name, 0, 1000, lambda x: x)
        cv.createTrackbar('Canny_2', window_name, 0, 1000, lambda x: x)
        cv.createTrackbar('Erode', window_name, 1, 1000, lambda x: x)
        cv.createTrackbar('Dilatate', window_name, 1, 1000, lambda x: x)


        image_num = 1
        canny_1 = 310
        canny_2 = 181
        erode = 1
        dilatate = 1
        while True:
            img = self.img_masked[image_num]
            img_gray = self.img_masked_gray[image_num]
            img_ori = self.img_masked[image_num]
            img_gray = cv.erode(img_gray, np.ones((erode,erode), np.uint8), iterations=2)
            img_gray = cv.dilate(img_gray, np.ones((dilatate,dilatate), np.uint8))
            # cv.imshow(window_name, self.resize(cv.Canny(img,canny_1,canny_2)))
            cv.imshow(window_name, self.resize(img_ori))
            thresh_platform = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
            apexes = []
            edges_platfrom = cv2.Canny(thresh_platform, canny_1, canny_2)
            lines_platform = cv2.HoughLinesP(edges_platfrom, 2, np.pi / 180, 120, minLineLength=erode, maxLineGap=dilatate)
            tmp = img_ori.copy()
            # create copy of image

            for line in lines_platform:
                x1, y1, x2, y2 = line[0]
                cv2.line(tmp, (x1, y1), (x2, y2), (0, 255, 0), 5)

                apexes.append([x1, y1])
                apexes.append([x2, y2])
            cv.imshow("Lines", self.resize(tmp))
            image_num = cv.getTrackbarPos('Image', window_name)
            # canny_1 = cv.getTrackbarPos('Canny_1', window_name)
            # canny_2 = cv.getTrackbarPos('Canny_2', window_name)
            erode = cv.getTrackbarPos('Erode', window_name)
            dilatate = cv.getTrackbarPos('Dilatate', window_name)

            key_code = cv.waitKey(10)
            if key_code == 27:
                break

        cv.destroyAllWindows()
        return
    def mask_tuning(self):
        import cv2 as cv
        max_value = 255
        max_value_H = 360//2
        low_H = 0
        high_H = 180
        low_S = 0
        high_S = 82
        low_V = 60
        high_V = max_value
        window_capture_name = 'Video Capture'
        window_detection_name = 'Object Detection'
        low_H_name = 'Low H'
        low_S_name = 'Low S'
        low_V_name = 'Low V'
        high_H_name = 'High H'
        high_S_name = 'High S'
        high_V_name = 'High V'
        image_num = 0
        def on_low_H_thresh_trackbar(val):
            nonlocal low_H
            nonlocal high_H
            low_H = val
            low_H = min(high_H-1, low_H)
            cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
        def on_high_H_thresh_trackbar(val):
            nonlocal low_H
            nonlocal high_H
            high_H = val
            high_H = max(high_H, low_H+1)
            cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
        def on_low_S_thresh_trackbar(val):
            nonlocal low_S
            nonlocal high_S
            low_S = val
            low_S = min(high_S-1, low_S)
            cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
        def on_high_S_thresh_trackbar(val):
            nonlocal low_S
            nonlocal high_S
            high_S = val
            high_S = max(high_S, low_S+1)
            cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
        def on_low_V_thresh_trackbar(val):
            nonlocal low_V
            nonlocal high_V
            low_V = val
            low_V = min(high_V-1, low_V)
            cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
        def on_high_V_thresh_trackbar(val):
            nonlocal low_V
            nonlocal high_V
            high_V = val
            high_V = max(high_V, low_V+1)
            cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
        def on_image_num_thresh_trackbar(val):
            nonlocal image_num
            image_num = val
            cv.setTrackbarPos("Image", window_detection_name, image_num)
        cv.namedWindow(window_capture_name)
        cv.namedWindow(window_detection_name)
        cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
        cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
        cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
        cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
        cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
        cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
        cv.createTrackbar("Image", window_detection_name , 0, len(self.img_color_list)-1, on_image_num_thresh_trackbar)

        while True:           
            image = self.img_color_list[image_num]
            width = int(image.shape[1] * 20 / 100)
            height = int(image.shape[0] * 20 / 100)
            dim = (width, height)
            frame = cv.resize(image, dim, interpolation=cv.INTER_AREA)
            if frame is None:
                break
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            frame_threshold = cv.bitwise_not(frame_threshold)
            final = cv.bitwise_and(frame, frame, mask=frame_threshold)
            
            cv.imshow(window_capture_name, frame)
            cv.imshow('Final', final)

            
            

            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break

if __name__ == "__main__":
    start_time = time.time()
    win = lab()
    print("--- %s seconds ---" % (time.time() - start_time))
    win.tuning()
   