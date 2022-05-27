import os

import cv2 as cv
from cv2 import COLOR_BGR2HLS
import numpy as np

from time import time as t_now

from pathlib import Path
from scipy.spatial import distance
from threading import Thread
from cv2 import imread, imshow, inRange, cvtColor, COLOR_BGR2HSV, bitwise_and
from imutils import contours, grab_contours, perspective, is_cv2
from skimage.metrics import structural_similarity 


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

    def __init__(self):
        
        self.images = {}
        self.elements = []
        self.result = {}

        threads = [ThreadWithReturnValue(target=self.load_shapes, args=(i ,)) for i in ["A","B","C","D","E"]]
        # threads = [ThreadWithReturnValue(target=self.load_shapes, args=(i ,)) for i in ["A","B"]]

        _ = [thread.start() for thread in threads]
        self.shapes = [thread.join() for thread in threads]
        self.load_images()
        self.crop_elements()
    
    @staticmethod
    def load_shapes(id): return [imread(os.path.join("dane",str(id),str(i)+".jpg"),0) for i in range(5)]

    def load_images(self,filespath = "train"):
        # images_dir = Path(args.images_dir)
        images_dir = Path(filespath)

        # results_file = Path(args.results_file)
        images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
        for image_path in images_paths:
            image = imread(str(image_path))
            if image is None:
                print(f'Error loading image {image_path}')
                continue
            img = image.copy()
            hsv = cvtColor(img, COLOR_BGR2HSV)
            hls = cvtColor(img, COLOR_BGR2HLS)

            # white = inRange(hsv, (27,0,172), (127,51,255))
            white1 = inRange(hsv, (7,0,170), (255,255,189))

            # yellow_green_red_blue = inRange(hsv, (0,90,0), (179,255,255))
            colors = inRange(hls, (0,57,0), (255,232,68))
            green = inRange(hls, (34,0,0), (131,93,255))
            colors = 255 - colors
            masks = colors + green +white1
            final = bitwise_and(img,img, mask= masks)
            # save image with hour
            self.images[image_path.name] = (image, final)
    
    def crop_elements(self):
        def midpoint(ptA, ptB): return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
        for img in self.images:
            res_tmp = {}
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
                    if res_tmp.get(shape) == None:
                        res_tmp[shape] = 1
                    else:
                        res_tmp[shape] = res_tmp[shape]+1

                    cv.putText(self.images[img][0], str(shape),(int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            cv.imwrite(f'{img}_{t_now()}.jpg', self.images[img][0])

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
        return id_tmp+1, max_tmp
    
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
    
    @staticmethod
    def resize(image = None,scale_percent :int =30):

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)
    
    def new_(self):

        window_name = "New"
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.images)-1, lambda x: x)
        images = [self.resize(self.images[img][0]) for img in self.images]
        image_num = 1
        while True:
            img_color = images[image_num]            
            imshow(window_name,img_color)
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
    print(win.result)
    win.new_()



# import argparse
# import json
# from pathlib import Path

# import cv2

# from processing.utils import perform_processing

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('images_dir', type=str)
#     parser.add_argument('results_file', type=str)
#     args = parser.parse_args()

#     images_dir = Path(args.images_dir)
#     results_file = Path(args.results_file)

#     images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
#     results = {}
#     for image_path in images_paths:
#         image = cv2.imread(str(image_path))
#         if image is None:
#             print(f'Error loading image {image_path}')
#             continue

#         results[image_path.name] = perform_processing(image)

#     with results_file.open('w') as output_file:
#         json.dump(results, output_file, indent=4)


# if __name__ == '__main__':
#     main()