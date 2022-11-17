import numpy as np
import cv2
import torch

from transformers import YolosFeatureExtractor, YolosForObjectDetection,  AutoFeatureExtractor, AutoModelForObjectDetection
from detect_info import DetectInfo
from detector import Detector

class YoloDetector(Detector):
    def __init__(self, th = 0.9, verbose=False):
        self._feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
        self._model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

        # self._feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny")
        # self._model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        self._th = th
        self._verbose = verbose
        
    def detect(self, frame: np.ndarray) -> list[DetectInfo]:
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

                d = DetectInfo(obj_label, tuple(box), score)
                detect_infos.append(d)
                
        return detect_infos
        
    
