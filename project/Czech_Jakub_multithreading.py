import cv2 as cv
import numpy as np


from threading import Thread
from os import listdir
from cv2 import imread, imshow, inRange, cvtColor, COLOR_BGR2HSV
from scipy.spatial import distance
from imutils import contours, grab_contours, perspective, is_cv2
from skimage.metrics import structural_similarity 
from time import time as t_now
import os
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
max_tmp = 0
id_tmp = 0
class Project:

    def __init__(self):
        self.images = []
        self.masked_img = []
        self.elements = []

        self.result = {}
        self.marked_img = []
        threads = []
        for i in ["A","B","C","D","E"]:
            threads.append(ThreadWithReturnValue(target=self.load_shapes, args=(i ,)))
        for thread in threads:
            thread.start()
        self.shapes = []
        for thread in threads:
            self.shapes.append(thread.join())

        self.load_images()
        self.mask_images()
        self.crop_elements()
    
    @staticmethod
    def load_shapes(id):
        shapes = []
        
        for i in range(5):

            img =imread(os.path.join("dane/",str(id),str(i)+".jpg"))
            shapes.append(img)
        return shapes

    def load_images(self):
        for file in listdir("SRC/train"):
            if file.endswith(".jpg"):
                img= imread("SRC/train/" + file) 
                self.images.append(img)
    
    def mask_images(self):
        for img in self.images:
            hsv = cvtColor(img, COLOR_BGR2HSV)
            white = inRange(hsv, (27,0,172), (127,51,255))
            yellow_green_red_blue = inRange(hsv, (0,90,0), (179,255,255))
            masks = yellow_green_red_blue + white
            final = cv.bitwise_and(img,img, mask= masks)
            self.masked_img.append(final)
    
    def crop_elements(self):
        def midpoint(ptA, ptB): return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
      
        
        for id,img in enumerate(self.masked_img):
            res_tmp = {}
            cnts = self.get_cnts(img)
            for c in cnts:
                if cv.contourArea(c) > 150000 or cv.contourArea(c) < 10000:
                    continue   
                box = self.get_box(c)    
                (tl, tr, br, bl) = box
                img1 = self.four_point_transform(img, box)

                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)      

                dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
                shape, score = self.compare_images(img1,dA,dB)
                if score > 0.65:
                    if res_tmp.get(shape) == None:
                        res_tmp[shape] = 1
                    else:
                        res_tmp[shape] = res_tmp[shape]+1
                
                    # cv.putText(img, str(shape),(int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            self.result[id] = res_tmp
    
    @staticmethod
    def get_box(c):
        box = cv.minAreaRect(c)
        box = cv.BoxPoints(box) if is_cv2() else cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        return box

    @staticmethod
    def get_cnts(img):
        gray = cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)
        edged = cv.Canny(gray, 50, 100)
        edged = cv.dilate(edged, None, iterations=1)
        edged = cv.erode(edged, None, iterations=1)
        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        return cnts
    
    def compare_images(self, img, h, w):

        shapes = [imread("1/7.jpg",0),imread("2/190.jpg",0),imread("3/14.jpg",0),imread("4/26.jpg",0),imread("5/2.jpg",0)]
        test_first_list = ThreadWithReturnValue(target=self.check_shapes, args=(shapes,img , h ,w ,))

        shapes = [imread("1/18.jpg",0),imread("2/188.jpg",0),imread("3/115.jpg",0),imread("4/144.jpg",0),imread("5/359.jpg",0)]
        test_second_list = ThreadWithReturnValue(target=self.check_shapes, args=(shapes,img , h ,w ,))
        
        test_first_list.start()
        test_second_list.start()

        id_tmp_1, max_tmp_1 = test_first_list.join()
        id_tmp_2, max_tmp_2 = test_second_list.join()
        if max_tmp_1 > max_tmp_2:
            return id_tmp_1+1, max_tmp_1
        elif max_tmp_2 > max_tmp_1:
            return id_tmp_2+1, max_tmp_2
    
    def check_shapes(self,shapes, img, h, w):
        max_tmp = 0
        id_tmp = 0
   
        for id, shape in enumerate(shapes):
            score = self.check_shape(img,shape,id)
            if id == 0 and (h/w > 2 or w/h > 2) :
                    score += 0.2
            if id == 3 and (h/w > 2 or w/h > 2) and score > 0.7:
                    score -= 0.6
            if ( id == 0 or id == 3) and (h < 100 or w < 100) and score > 0.7:
                score -= 0.7
            
            if score > 0.9:
                return id, score
            elif score > max_tmp:
                max_tmp = score
                id_tmp = id
        return id_tmp, max_tmp

    @staticmethod
    def check_shape(img_ori,shape, id):
        max_score = 0
        for i in range(4):
            img = img_ori
            shape = cv.rotate(shape, cv.ROTATE_90_CLOCKWISE)
            shape_copy = shape.copy()
            img = inRange(img, (0,0,0), (0,0,15))
            width = int(shape_copy.shape[1])
            height = int(shape_copy.shape[0])
            dim = (width, height)
            img =  cv.resize(img, dim, interpolation=cv.INTER_AREA)
            (score, diff) = structural_similarity(img, shape_copy, full=True)
            if score > 0.95:
                return score
            max_score = score
        return max_score
      
    @staticmethod
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
                M = cv.getPerspectiveTransform(pts, dst)
                warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
                return warped

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
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)
    
    def new_(self):
        window_name = "New"
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.images)-1, lambda x: x)
        image_num = 1
        while True:
            img_color = self.masked_img[image_num]
            img_marked = self.images[image_num]
            imshow(window_name,np.hstack((self.resize(img_marked), self.resize(img_color))))
            image_num = cv.getTrackbarPos('Image', window_name)
            key_code = cv.waitKey(10)

            if key_code == 27:
                break

        cv.destroyAllWindows()
        return

if __name__ == "__main__":
    start_time = t_now()
    print("Start time: ", start_time)
    win = Project()
    print("--- %s seconds ---" % (t_now() - start_time))
    # win.new_()
    print(win.result)