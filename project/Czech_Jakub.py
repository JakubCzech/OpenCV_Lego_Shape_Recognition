import cv2 as cv,cv2
from os import listdir

from cv2 import imread, imshow, inRange
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils import contours, grab_contours, perspective
import numpy as np
import imutils
import cv2
from skimage.metrics import structural_similarity as compare_ssim

class Project:

    def __init__(self):
        self.images = []
        self.masked_img = []
        self.elements = []

        self.result = {}
        self.marked_img = []

        self.load_images()
        self.mask_images()
        self.crop_elements()
       
    def load_images(self):
        for file in listdir("SRC/train"):
            if file.endswith(".jpg"):
                img= imread("SRC/train/" + file) 
                self.images.append(img)
    
    def mask_images(self):
        for img in self.images:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            white = cv2.inRange(hsv, (27,0,172), (127,51,255))
            yellow_green_red_blue = cv2.inRange(hsv, (0,90,0), (179,255,255))
            masks = yellow_green_red_blue + white
            final = cv2.bitwise_and(img,img, mask= masks)
            self.masked_img.append(final)
    
    def crop_elements(self):
        def midpoint(ptA, ptB): return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
      
        
        for id,img in enumerate(self.masked_img):
            res_tmp = {}
            cnts = self.get_cnts(img)
            for c in cnts:
                if cv2.contourArea(c) > 150000 or cv2.contourArea(c) < 10000:
                    continue   
                box = self.get_box(c)    
                (tl, tr, br, bl) = box
                img1 = self.four_point_transform(img, box)

                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)      

                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                shape, score = self.compare_images(img1,dA,dB)
                if score > 0.65:
                    if res_tmp.get(shape) == None:
                        res_tmp[shape] = 1
                    else:
                        res_tmp[shape] = res_tmp[shape]+1
                
                    # cv2.putText(img, str(shape),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,5, (255, 255, 255), 2)
            self.result[id] = res_tmp
    
    @staticmethod
    def get_box(c):
        box = cv2.minAreaRect(c)
        box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        return box

    @staticmethod
    def get_cnts(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        return cnts
    
    def compare_images(self, img, h, w):
        max_tmp = 0
        id_tmp = 0
        shapes = [imread("1/7.jpg",0),imread("2/190.jpg",0),imread("3/14.jpg",0),imread("4/26.jpg",0),imread("5/2.jpg",0)]
        for id, shape in enumerate(shapes):
            score = self.check_shape(img,shape,id)
            if id == 0 and (h/w > 2 or w/h > 2) :
                    score += 0.2
            if id == 3 and (h/w > 2 or w/h > 2) and score > 0.7:
                    score -= 0.6
            if ( id == 0 or id == 3) and (h < 100 or w < 100) and score > 0.7:
                score -= 0.7
            
            if score > 0.9:
                return id+1, score
            elif score > max_tmp:
                max_tmp = score
                id_tmp = id
        shapes = [imread("1/18.jpg",0),imread("2/188.jpg",0),imread("3/115.jpg",0),imread("4/144.jpg",0),imread("5/359.jpg",0)]
        for id, shape in enumerate(shapes):
            score = self.check_shape(img,shape,id)
            if id == 0 and (h/w > 2 or w/h > 2) :
                    score += 0.2
            if id == 3 and (h/w > 2 or w/h > 2) and score > 0.7:
                    score -= 0.4
            if ( id == 0 or id == 3) and (h < 100 or w < 100) and score > 0.7:
                score -= 0.7
            
            if score > 0.9:
                return id+1, score
            elif score > max_tmp:
                max_tmp = score
                id_tmp = id
        return id_tmp+1, max_tmp

    @staticmethod
    def check_shape(img_ori,shape, id):
        max_score = 0
        for i in range(4):
            img = img_ori
            shape = cv2.rotate(shape, cv2.ROTATE_90_CLOCKWISE)
            shape_copy = shape.copy()
            img = inRange(img, (0,0,0), (0,0,15))
            width = int(shape_copy.shape[1])
            height = int(shape_copy.shape[0])
            dim = (width, height)
            img =  cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            (score, diff) = compare_ssim(img, shape_copy, full=True)
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
                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
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
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    def new_(self):
        window_name = "New"
        cv.namedWindow(window_name)
        cv.createTrackbar('Image', window_name, 0, len(self.images)-1, lambda x: x)
        image_num = 1
        while True:
            img_color = self.masked_img[image_num]
            img_marked = self.images[image_num]
            cv2.imshow(window_name,np.hstack((self.resize(img_marked), self.resize(img_color))))
            image_num = cv.getTrackbarPos('Image', window_name)
            key_code = cv.waitKey(10)

            if key_code == 27:
                break

        cv.destroyAllWindows()
        return

if __name__ == "__main__":
    start_time = time.time()
    win = Project()
    print("--- %s seconds ---" % (time.time() - start_time))
    # win.new_()
    print(win.result)