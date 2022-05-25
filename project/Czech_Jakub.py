from tkinter.tix import IMAGE
from typing_extensions import Self
import cv2 as cv,cv2
from os import listdir

from cv2 import imread
import numpy as np
import time
import imutils 
from scipy.spatial import distance as dist
from imutils import contours, grab_contours, perspective
import numpy as np
import imutils
import cv2
from skimage.metrics import structural_similarity as compare_ssim


class Element:
    def __init__(self,w,h,box):
        self.type = "Element"
        if 380 <w< 500 and 120<h<180:
            self.type = "S"
        if 380 <h< 500 and 120<w<180:
            self.type = "S"
        if 240 <w< 280 and 300<h<340:
            self.type = "A"
        if 240 <h< 280 and 300<w<340:
            self.type = "A"
        self.img = None
        self.box = box
    def __str__(self):
        return "Name: " + self.name + " Color: " + self.color + " Position: " + str(self.position)

class lab:
        
    def __init__(self):
        self.img_masked = []
        self.elements = []
        self.marked_img = []
        self.load_images()
       
    def load_images(self):
        for file in listdir("SRC/train"):
            if file.endswith(".jpg"):
                img_color = cv.imread("SRC/train/" + file)  
                img_masked = self.get_color_shape(img_color)              
                self.img_masked.append(img_masked)   
                mark_img , elements =self.create_elements(img_masked)
                self.marked_img.append(mark_img)
                self.elements.append(elements)

    @staticmethod
    def create_elements(image):
        
        def midpoint(ptA, ptB): return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        orig = image.copy()
        elements = []
        shape = imread("2/185.jpg",0)

        def four_point_transform(image, pts):

                (tl, tr, br, bl) = pts
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                return warped
        for id,c in enumerate(cnts):
            if cv2.contourArea(c) > 150000 or cv2.contourArea(c) < 10000:
                continue       
            box = cv2.minAreaRect(c)

            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            img = four_point_transform(orig, box)
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)        
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            blue_mask = cv.inRange(img, (0,0,0), (0,0,15))
            width = int(blue_mask.shape[1])
            height = int(blue_mask.shape[0])
            dim = (width, height)
            shape=  cv2.resize(shape, dim, interpolation=cv2.INTER_AREA)
            (score, diff) = compare_ssim(blue_mask, shape, full=True)
            cv.imshow(str(diff), shape)
            cv.imshow(str(score), blue_mask)
            cv.waitKey(0)
            cv.destroyAllWindows()
            black = sum(sum(blue_mask))



            # compare blue_mask with shape from shapes/


            # if dA/dB > 2 or dB/dA > 2:
                
            #     if 350 <dA< 550 and 100<dB<200:
            #         cv2.putText(orig, "S",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            #     elif 350 <dB< 550 and 100<dA<200:
            #         cv2.putText(orig, "S",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            #     # else:
            #     #     cv2.putText(orig, "{:.1f}px".format(dA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)
            #     #     cv2.putText(orig, "{:.1f}px".format(dB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)
            # elif (dA/dB > 6/10 and dA/dB < 15/10) or (dB/dA > 6/10 and dB/dA < 14/10):
            #     if 200 <dA< 330 and 200<dB<290 and black <20000:
            #         cv2.putText(orig, "A",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            #     elif 200 <dB< 330 and 200<dA<290 and black <20000:
            #         cv2.putText(orig, "A",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            #     else :
            #         cv2.putText(orig, "{:.1f}px".format(black),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)
                    # imshow(str(sum(sum(blue_mask))),blue_mask)
                    # imwrite(str(sum(sum(blue_mask)))+".jpg",blue_mask)
                    # cv.waitKey(0)
                # cv2.putText(orig, "{:.1f}px".format(dA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)
                # cv2.putText(orig, "{:.1f}px".format(dB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)


            # elif 240 <dA< 280 and 300<dB<360:
            #     cv2.putText(orig, "A",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            # elif 240 <dB< 280 and 300<dA<360:
            #     cv2.putText(orig, "A",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            # elif 160 <dA< 260 and 200<dB<300:
            #     cv2.putText(orig, "B",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            # elif 160 <dB< 260 and 200<dA<300:
            #     cv2.putText(orig, "B",(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            # else:
            #     # cv2.putText(orig, "{:.1f}px".format(dA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)
            #     # cv2.putText(orig, "{:.1f}px".format(dB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)
            #     # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            #     # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            #     # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
            #     pass
            elements.append(Element(dA,dB,box))    
        return orig ,elements

    @staticmethod
    def resize(image = None,scale_percent :int =30):
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
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (27,0,172), (127,51,255))
        yellow_green_red_blue = cv2.inRange(hsv, (0,90,0), (179,255,255))
        masks = yellow_green_red_blue + white
        return cv2.bitwise_and(image,image, mask= masks)


    def new_(self):
        window_name = "New"
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.elements)-1, lambda x: x)
        image_num = 1
        while True:
            img_color = self.img_masked[image_num]
            img_marked = self.marked_img[image_num]
            elements = self.elements[image_num]
            cv2.imshow(window_name,np.hstack((self.resize(img_marked), self.resize(img_color))))
            counter = 0
            counter1 = 0

            for e in elements:
                if e.type == "S":
                    counter+=1
                if e.type == "A":
                    counter1+=1
            print(f'S {counter} A = {counter1}')
            image_num = cv.getTrackbarPos('Image', window_name)
            key_code = cv.waitKey(10)

            if key_code == 27:
                break

        cv.destroyAllWindows()
        return

   

if __name__ == "__main__":
    start_time = time.time()
    win = lab()
    print("--- %s seconds ---" % (time.time() - start_time))
    win.new_()
   