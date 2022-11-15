import tensorflow as tf
import numpy as np
import cv2
import time

from detector import Detector
from detect_dert import DetrDetector
from detect_yolo import YoloDetector
from utils import draw_to_cv2


def run_webcam(cap: cv2.VideoCapture, detector: Detector):
    cnt = 0
    items = []
    
    while(True):
        ret, frame = cap.read()

        if(cnt % 5):
            items = detector.detect(frame)
            
        for item in items:
            if(item.label == "carrot"):
                frame = draw_to_cv2(frame, item.box, item.label, False, False)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt += 1

    # When everything done, release the capture
   
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 5)

    # detector = DetrDetector()
    detector = YoloDetector(verbose=True)
    run_webcam(cap, detector)
    
    cap.release()
    cv2.destroyAllWindows()
