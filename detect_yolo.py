import numpy as np
import cv2
import torch

from transformers import YolosFeatureExtractor, YolosForObjectDetection,  AutoFeatureExtractor, AutoModelForObjectDetection
from detect_info import DetectInfo
from detector import Detector

default_model = 'hustvl/yolos-small'

class YoloDetector(Detector):
    """
    th: detectする閾値 (default 0.8)
    resize_rate: detect時の、入力画像をresizeする比率 (default 0.5)
    """
    def __init__(self, model = default_model, th = 0.8, resize_rate = 1.0, verbose=False):
        self._feature_extractor = YolosFeatureExtractor.from_pretrained(default_model)
        self._model = YolosForObjectDetection.from_pretrained(model)

        # self._feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny")
        # self._model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        self._th = th
        self._verbose = verbose
        self._resize_rate = resize_rate
        
    def detect(self, frame: np.ndarray) -> list[DetectInfo]:
        frame = cv2.resize(frame, dsize=None, fx=self._resize_rate, fy=self._resize_rate)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # model_input = processing.normalized_images(model_input, config)

        inputs = self._feature_extractor(images=image, return_tensors="pt")
        outputs = self._model(**inputs)
        
        target_sizes = torch.tensor([image.shape[:2]])
        results = self._feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    
        detect_infos = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # let's only keep detections with score > 0.9
            if score > self._th:
                obj_label = self._model.config.id2label[label.item()]
                score = round(score.item(), 3)

                if(self._verbose):
                    print("Detected {} with confidence {} at location {}"
                          .format(obj_label,score, box))
                box = [ v / self._resize_rate for v in box]
                if(obj_label == "turnip"):
                    d = DetectInfo(obj_label, tuple(box), score)
                    detect_infos.append(d)                
        return detect_infos
        
    
