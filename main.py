"""
run:
python main.py


カメラIDの取得
mac: ffmpeg -f avfoundation -list_devices true -i ""
"""



import tensorflow as tf
import numpy as np
import cv2
import time
import os

import queue
import numpy as np

from detector import Detector
from detect_dert import DetrDetector
from detect_yolo import YoloDetector
from detect_info import DetectInfo
from utils import draw_to_cv2, cal_size, set_arrow, crop, nparr_to_img, cal_size_using_bg, nms

from concurrent import futures

import queue

WIDTH=1280
HEIGHT=720

FPS=1
WINDOW_SIZE_RATE=1.0

def a_detect(detector, frame):
    return detector.detect(frame)
    
def async_run_webcam(cap: cv2.VideoCapture, detector: Detector):
    q = queue.Queue(1)
    executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="thread")
    future = None
    items = []
    draw_cnt = 0
    
    while(True):
        ret, frame = cap.read()
        if(q.empty()):
            q.put("running")
            _frame = np.copy(frame)
            future = executor.submit(a_detect, detector, _frame)
            
        if (future is not None) and (future.done()) and (q.full()):
            items = future.result()
            if(len(items) != 0):
                items = nms(items, 0.7)

            for item in items:
                print("crop")
                (x1, y1, x2, y2) = item.box
                img = np.copy(frame)
                cropped = crop(img, int(x1), int(y1), int(x2), int(y2))
                print("cal size")
                # size = cal_size(cropped)
                size = cal_size_using_bg(cropped)
                item.set_size(size)       
            q.get()

        if len(items) > 1:
            set_arrow(items)
            
        set_arrow(items)
        for item in items:
            label = "size:{}".format(item.size)
            frame = draw_to_cv2(frame, item.box, label, item.left_arrow, item.right_arrow)
            draw_cnt += 1
            
        if(draw_cnt > 50):
            items = []
            draw_cnt = 0
            
        if(frame is not None):
            frame = cv2.resize(frame, dsize=None, fx=WINDOW_SIZE_RATE, fy=WINDOW_SIZE_RATE)
            cv2.imshow('frame', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if(future is not None):
                future.cancel()
            break

    
def run_webcam(cap: cv2.VideoCapture, detector: Detector):
    cnt = 0
    items = []
    q = queue.Queue(1)
        
    while(True):
        ret, frame = cap.read()
                        
        if(cnt % 5):
            items = detector.detect(frame)

        # for item in items:
        #     img = np.copy(frame)            
        #     x1,y1,x2,y2 = item.box
        #     img = crop(img, int(x1), int(y1), int(x2), int(y2))
        #     _tmp = nparr_to_img(img)
        #     _tmp.show()
        #     os.exit(1)
            
        #     cropped = frame.crop(item.box)
        #     size = cal_size(cropped)
        #     item.set_size(size)
        # set_arrow(items)
            
        for item in items:            
            frame = draw_to_cv2(frame, item.box, item.label, False, False)
            
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt += 1

    # When everything done, release the capture


def test_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
if __name__ == "__main__":
    # test_webcam()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # detector = DetrDetector()
    detector = YoloDetector(th = 0.8, resize_rate=0.5, verbose=True)
    
    # run_webcam(cap, detector)
    async_run_webcam(cap, detector)
    
    cap.release()
    cv2.destroyAllWindows()
