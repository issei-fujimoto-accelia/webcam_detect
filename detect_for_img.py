from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from utils import draw_to_cv2, cal_size, set_arrow, img_to_nparr, crop, cal_size_using_bg, nms, nparr_to_img
from detect_yolo import YoloDetector

def main():
    # image = Image.open("./images/kabu3.jpeg")
    image = Image.open("./images/kabu_webcam.png")

    detector = YoloDetector(th = 0.8, resize_rate=0.5, verbose=True)
    image = img_to_nparr(image)
    items = detector.detect(image)
    items = nms(items, 0.7)

    print("nms")
    items = nms(items, 0.7)
    print("items", items)
    for item in items:
        print("crop")
        (x1, y1, x2, y2) = item.box
        image = np.copy(image)
        cropped = crop(image, int(x1), int(y1), int(x2), int(y2))
        print("cal size")
        size = cal_size_using_bg(cropped)
        size2 = cal_size(cropped)
        print("size", size, size2)
        item.set_size(size)
        
    if len(items) != 0:
        items = set_arrow(items)


    print("draw")
    for item in items:
        label = "{}, score:{}, size: {}".format(item.label, item.score, item.size)
        image = draw_to_cv2(image, item.box, label, item.left_arrow, item.right_arrow)
    image = nparr_to_img(image)
    image.show()


if __name__ == "__main__":
    main()
