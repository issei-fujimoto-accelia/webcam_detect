from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import argparse
import random

from utils import draw_to_cv2, cal_size, set_arrow, img_to_nparr, crop, cal_size_using_bg, nms, nparr_to_img

from utils import ArrangementArrow
from detect_yolo import YoloDetector

# from predict_distance import midas
import predict_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help="input image for prediction", required=True)
    parser.add_argument('-m', '--model', help="model")
    args = parser.parse_args() 

    input_img = args.image
    model = args.model
    # image = Image.open("./images/kabu3.jpeg")
    # image = Image.open("./images/kabu_webcam.png")
    # image = Image.open("./dataset/raw_cam/fujimoto_000024.jpg")
    # image = Image.open("./images/IMG_0635.jpg")
    image = Image.open(input_img)
    image = img_to_nparr(image)
    

    (h, w, _) = image.shape
    arrangement = ArrangementArrow(w, h)    
    detector = YoloDetector(model = model, th = 0.8, resize_rate=0.5, verbose=True)
        
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
        
    # if len(items) != 0:
    #     items = set_arrow(items)

    if len(items) > 0:
        arrangement.set_arrow(image, items[0])

    print("draw")
    for item in items:
        label = "{}, score:{}, size: {}".format(item.label, item.score, item.size)
        image = draw_to_cv2(image, item.box, label, item.left_arrow, item.right_arrow)
    image = nparr_to_img(image)
    save_file = "./tmp_{}.jpg".format(random.randint(0, 1000))
    image.save(save_file)
    print("saved image...", save_file)    
    # image.show()

if __name__ == "__main__":
    # test()
    main()
