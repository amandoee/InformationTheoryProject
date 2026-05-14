"""Detection module for identifying objects in images using YOLO."""

import cv2
import numpy as np
from typing import List, Tuple


class ObjectDetector:
    """Wrapper class for object detection."""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):

        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.net = None
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load YOLO model from weights file."""
        try:
            # This is a placeholder for YOLO model loading
            # You can use ultralytics YOLOv8 or OpenCV's DNN module
            print(f"Loading model from {model_path}")
            # Example: self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    
    def detect_with_ultralytics(self, image: np.ndarray, model_name: str = "yolov8n.pt") -> List[Tuple]:

        try:
            from ultralytics import YOLO
            
            model = YOLO(model_name)
            results = model(image, conf=self.conf_threshold, verbose=False)
            
            bounding_boxes = []
            height, width, _ = image.shape
            
            for result in results:
                for box in result.boxes:
                    # Get normalized coordinates from YOLO
                    x_center = float(box.xywhn[0][0])
                    y_center = float(box.xywhn[0][1])
                    box_width = float(box.xywhn[0][2])
                    box_height = float(box.xywhn[0][3])
                    class_id = int(box.cls[0])
                    
                    bounding_boxes.append((class_id, x_center, y_center, box_width, box_height))
            
            return bounding_boxes
        
        except ImportError:
            print("Ultralytics YOLO not installed. Install with: pip install ultralytics")
            return []
        except Exception as e:
            print(f"Error during detection: {e}")
            return []


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def save_image(image: np.ndarray, output_path: str):
    cv2.imwrite(output_path, image)
