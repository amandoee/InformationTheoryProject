"""Semantic description module using CLIP for scene understanding."""

import numpy as np
import torch
from PIL import Image


class SemanticDescriber:
    """Generate semantic scene descriptions using CLIP."""
    
    def __init__(self, device: str = None):
        """
        Initialize CLIP-based semantic describer.
        
        Args:
            device: Device to use ("cpu", "cuda", or None for auto-detect)
        """
        self.device = device or self._auto_detect_device()
        self.processor = None
        self.model = None
        self._initialize_model()
    
    def _auto_detect_device(self) -> str:
        """Auto-detect available device."""
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def _initialize_model(self):
        """Initialize CLIP model for scene classification."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("Loading CLIP model for scene understanding...")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"CLIP model loaded successfully on {self.device}")
        
        except ImportError as e:
            print(f"Error: Could not load CLIP model: {e}")
            print("Install with: pip install transformers torch pillow")
            raise
    
    def describe_environment(self, image: np.ndarray) -> str:
        """
        Generate environment and scene description using CLIP.
        
        Args:
            image: Image as numpy array (BGR or RGB)
            
        Returns:
            Environment description
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[..., ::-1]  # BGR to RGB
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Environment and setting focused descriptions
            environment_descriptions = [
                "indoor environment",
                "outdoor environment",
                "urban setting",
                "natural landscape",
                "bright and well-lit",
                "dark or dimly lit",
                "daytime scene",
                "nighttime scene",
                "close-up view",
                "wide landscape view",
                "minimalist background",
                "complex detailed background",
            ]
            
            inputs = self.processor(text=environment_descriptions, images=pil_image, 
                                  return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            best_idx = logits_per_image[0].argmax().item()
            
            return f"Environment: {environment_descriptions[best_idx]}"
        
        except Exception as e:
            print(f"Error generating description: {e}")
            raise


def generate_description_with_detections(image: np.ndarray, 
                                        bounding_boxes: list) -> str:
    """
    Generate scene description with object count.
    
    Args:
        image: Input image
        bounding_boxes: List of detected bounding boxes
        
    Returns:
        Description of environment + object count
    """
    describer = SemanticDescriber()
    environment_description = describer.describe_environment(image)
    
    num_objects = len(bounding_boxes)
    if num_objects > 0:
        return f"{environment_description} {num_objects} object(s) detected."
    else:
        return environment_description + " No objects detected."
