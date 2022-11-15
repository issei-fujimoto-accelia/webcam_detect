import abc
import cv2
import numpy as np

from detect_info import DetectInfo

class Detector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(frame: np.ndarray) -> list[DetectInfo]:
        raise NotImplementedError()

