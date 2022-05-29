import os

import cv2 as cv
import numpy as np
from scipy.spatial import distance
from threading import Thread
from cv2 import imread, inRange, cvtColor
from imutils import contours, grab_contours, perspective
from skimage.metrics import structural_similarity 
from json import dump

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class Project:

    def __init__(self, filespath = "train"):
        
        self.images = {}
        self.elements = []
        self.result = {}

        threads = [ThreadWithReturnValue(target=self.load_shapes, args=(i ,)) for i in ["A","B","C","D","E"]]
        _ = [thread.start() for thread in threads]
        self.shapes = [thread.join() for thread in threads]

        self.load_images(filespath)
        self.crop_elements()
    
    @staticmethod
    def load_shapes(id): return [imread(os.path.join("processing","dane",str(id),str(i)+".jpg"),0) for i in range(5)]

    def load_images(self,filespath):

        images_paths = sorted([image_path for image_path in filespath.iterdir() if image_path.name.endswith('.jpg')])
        for image_path in images_paths:
            image = imread(str(image_path))
            if image is None:
                print(f'Error loading image {image_path}')
                continue
            img = image.copy()
            hsv = cvtColor(img, cv.COLOR_BGR2HSV)
            hls = cvtColor(img, cv.COLOR_BGR2HLS)

            white1 = inRange(hsv, (7,0,170), (255,255,189))

            colors = inRange(hls, (0,57,0), (255,232,68))
            green = inRange(hls, (34,0,0), (131,93,255))
            colors = 255 - colors
            masks = colors + green +white1
            final = cv.bitwise_and(img,img, mask= masks)
            self.images[image_path.name] = (image, final)
    
    def crop_elements(self):
        def midpoint(ptA, ptB): return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
        for img in self.images:
            res_tmp = [0]*11
            img_tmp = self.images[img][1]
            cnts = self.get_cnts(img_tmp)
            for c in cnts:
                if cv.contourArea(c) > 150000 or cv.contourArea(c) < 10000:
                    continue   
                box = self.get_box(c)    
                (tl, tr, br, bl) = box
                img1 = self.four_point_transform(img_tmp, box)

                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)      

                dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
                shape, score = self.compare_images(img1,dA,dB)
                if score > 0.65:
                    res_tmp[shape] += 1
            self.result[img] = res_tmp
    
    def compare_images(self, img, h, w):   
        
        threads = [ThreadWithReturnValue(target=self.check_shapes, args=(shapes,img , h ,w ,)) for shapes in self.shapes]
        _ = [thread.start() for thread in threads]
        result = [thread.join() for thread in threads]
        
        id_tmp = 0
        max_tmp = 0
        for result in result:
            if result[1] > max_tmp:
                max_tmp = result[1]
                id_tmp = result[0]
        return id_tmp, max_tmp
    
    def check_shapes(self,shapes, img, h, w):
        max_tmp = 0
        id_tmp = 0
   
        for id, shape in enumerate(shapes):
            score = self.check_shape(img,shape)
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
    def check_shape(img_ori,shape):
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
    def get_box(c):
        box = cv.minAreaRect(c)
        box = cv.boxPoints(box)
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
    
    @staticmethod
    def resize(image = None,scale_percent :int =30):

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)
    
    def save_result(self, results_file):
        with results_file.open('w') as output_file:
            dump(self.result, output_file, indent=4)
       