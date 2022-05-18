import cv2 as cv,cv2
from os import listdir

class lab:
    img_color_list = []
    img_grey_list = []
    
    def __init__(self):
        self.load_images()
       

    def load_images(self):
        for file in listdir("SRC/train"):
            if file.endswith(".jpg"):
                img_color = cv.imread("SRC/train/" + file)
                img_grey = cv.imread("SRC/train/" + file, cv.IMREAD_GRAYSCALE)
                _, thresh1 = cv2.threshold(img_grey, 102, 255, cv2.THRESH_TOZERO)
                self.img_color_list.append(img_color)
                self.img_grey_list.append(thresh1)
    
    @staticmethod
    def resize(scale_percent :int =20, image = None):
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
    def canny(image,size1,size2):
        edges = cv.Canny(image, size1, size2)
        return edges

    def show_images(self):

        def size_callback(value):
            print(f'Size: {value}')
            return

        cv.namedWindow('Canny')

        cv.createTrackbar('Size1', 'Canny', 0, 1000, lambda x: x)
        cv.createTrackbar('Size2', 'Canny', 0, 1000, lambda x: x)
        cv.createTrackbar('Image', 'Canny', 0, len(self.img_grey_list)-1, lambda x: x)

        size1= 52
        size2 = 31
        image_num = 1
        
        while True:
            # prepare image
            img_grey = self.img_grey_list[image_num]
            img_color = self.img_color_list[image_num]
            
            cv.imshow('Clear', self.resize( image=img_color))
            cv.imshow('Canny', self.resize( image=self.canny(img_grey,size1,size2)))
            cv.imshow('Canny const', self.resize( image=self.canny(img_grey,6,100)))

            size1 = cv.getTrackbarPos('Size1', 'Canny')
            size2 = cv.getTrackbarPos('Size2', 'Canny')
            image_num = cv.getTrackbarPos('Image', 'Canny')

            key_code = cv.waitKey(10)
            if key_code == 27:
                break

        cv.destroyAllWindows()
        return



if __name__ == "__main__":
    win = lab()
    win.show_images()
    