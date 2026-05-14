"""Image reconstruction from subimages and semantic descriptions using inpainting."""

import json
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Optional


class ImageReceiver:
    """Reconstructs images from subimages and semantic environment descriptions."""
    
    def __init__(self, device: Optional[str] = None, lazy_load_model: bool = True):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ImageReceiver initialized on {self.device}")
        self.pipe = None
        self.lazy_load_model = lazy_load_model
        
        if not lazy_load_model:
            self._initialize_inpaint_model()
    
    def _initialize_inpaint_model(self):
        """Load Stable Diffusion inpainting model."""
        if self.pipe is not None:
            return  # Already loaded
            
        try:
            from diffusers import StableDiffusionInpaintPipeline
            model_id = "runwayml/stable-diffusion-inpainting"
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                safety_checker=None  # Disable safety checker for speed
            )
            self.pipe = self.pipe.to(self.device)
            
        except ImportError as e:
            self.pipe = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.pipe = None
    
    def load_semantic_metadata(self, semantic_json_path: str) -> Dict:

        with open(semantic_json_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def reconstruct_from_subimages(
        self,
        subimages_dir: str,
        semantic_json_path: str,
        output_path: str = "reconstructed.png",
        generate_background: bool = True,
        inference_steps: int = 4,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        
        # Load metadata
        metadata = self.load_semantic_metadata(semantic_json_path)
        height = metadata['original_height']
        width = metadata['original_width']
        description = metadata['semantic_description']
        
        print(f"Reconstructing image: {width}x{height}")
        print(f"Semantic description: {description}")
        print(f"Subimages to place: {metadata['num_subimages']}")
        
        # Create partial image with subimages placed
        partial_image = self._place_subimages(subimages_dir, metadata)
        
        # Create inpaint mask
        mask = self._create_inpaint_mask(partial_image)
        
        # Create object mask to preserve original subimages after inpainting
        object_mask = (partial_image.sum(axis=2) > 10).astype(np.uint8)  # Non-zero pixels
        
        # Generate background if requested
        if generate_background:
            reconstructed = self._generate_with_inpainting(
                partial_image, mask, description,
                inference_steps, guidance_scale
            )
            # Restore original subimages on top (to avoid inpainting artifacts on objects)
            reconstructed[object_mask > 0] = partial_image[object_mask > 0]
        else:
            print("Skipping background generation (disabled)")
            reconstructed = partial_image
        
        # Save result
        cv2.imwrite(output_path, reconstructed)
        print(f"Reconstructed image saved to {output_path}")
        
        return reconstructed
    
    def _place_subimages(self, subimages_dir: str, metadata: Dict) -> np.ndarray:
        
        height = metadata['original_height']
        width = metadata['original_width']
        
        # Create black canvas
        partial = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Place each subimage
        for i, subimg_info in enumerate(metadata['subimages']):
            x = subimg_info['x_position']
            y = subimg_info['y_position']
            subimg_height = subimg_info['height']
            subimg_width = subimg_info['width']
            
            # Try to find subimage file (could be named by index or class_id)
            subimg_paths = [
                os.path.join(subimages_dir, f"subimage_{i}.png"),
                os.path.join(subimages_dir, f"object_{i}.png"),
                os.path.join(subimages_dir, f"subimage_{i}.jpg"),
                os.path.join(subimages_dir, f"object_{i}.jpg"),
            ]
            
            subimg_path = None
            for path in subimg_paths:
                if os.path.exists(path):
                    subimg_path = path
                    break
            
            if subimg_path is None:
                print(f"  Warning: Could not find subimage {i} in {subimages_dir}")
                continue
            
            # Load and place subimage
            subimg = cv2.imread(subimg_path)
            if subimg is None:
                print(f"  Warning: Could not load {subimg_path}")
                continue
            
            # Ensure dimensions match
            h, w = subimg.shape[:2]
            
            # Place on canvas
            y_end = min(y + h, height)
            x_end = min(x + w, width)
            partial[y:y_end, x:x_end] = subimg[:y_end-y, :x_end-x]
            
            print(f"  Placed subimage {i} at ({x}, {y})")
        
        return partial
    
    def _create_inpaint_mask(self, image: np.ndarray) -> np.ndarray:

        # Convert to grayscale to detect black regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Black pixels: all channels near 0
        is_black = gray < 10  # Threshold for "black"
        
        # Create mask: white where black (areas to inpaint)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[is_black] = 255
        
        return mask
    
    def _generate_with_inpainting(
        self,
        partial_image: np.ndarray,
        mask: np.ndarray,
        description: str,
        inference_steps: int = 4,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Use Stable Diffusion inpainting to fill missing areas based on description.
        
        Args:
            partial_image: Image with subimages placed (rest black)
            mask: Binary mask of areas to inpaint
            description: Semantic description for prompt engineering
            inference_steps: Diffusion steps
            guidance_scale: Guidance scale
            
        Returns:
            Reconstructed image with filled background
        """
        print("  Generating background using AI inpainting...")
        
        # Lazy load model if needed
        if self.pipe is None:
            print("  Loading cached model...")
            self._initialize_inpaint_model()
        
        if self.pipe is None:
            print("  ✗ Inpainting model not available")
            print("  Skipping background generation")
            return partial_image
        
        try:
            # Stable Diffusion requires dimensions divisible by 8
            original_height, original_width = partial_image.shape[:2]
            
            # Round down to nearest multiple of 8
            adjusted_height = (original_height // 8) * 8
            adjusted_width = (original_width // 8) * 8
            
            # Resize if needed
            if adjusted_height != original_height or adjusted_width != original_width:
                print(f"  Resizing {original_width}x{original_height} → {adjusted_width}x{adjusted_height} (divisible by 8)")
                image_resized = cv2.resize(partial_image, (adjusted_width, adjusted_height))
                mask_resized = cv2.resize(mask, (adjusted_width, adjusted_height))
            else:
                image_resized = partial_image
                mask_resized = mask
            
            # Convert image to PIL
            image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask_resized)
            
            # Create prompt from semantic description
            prompt = self._create_inpainting_prompt(description)
            print(f"  Prompt: {prompt}")
            
            # Run inpainting
            print(f"  Running {inference_steps} diffusion steps...")
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    height=adjusted_height,
                    width=adjusted_width,
                )
            
            # Convert back to numpy BGR
            result_image = np.array(result.images[0])
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            
            # Resize back to original dimensions if needed
            if adjusted_height != original_height or adjusted_width != original_width:
                print(f"  Resizing back to original {original_width}x{original_height}")
                result_bgr = cv2.resize(result_bgr, (original_width, original_height))
            
            print("  ✓ Background generation complete")
            
            return result_bgr
            
        except Exception as e:
            print(f"  ✗ Error during inpainting: {e}")
            import traceback
            traceback.print_exc()
            print("  Returning partial image without inpainting")
            return partial_image
    
    def _create_inpainting_prompt(self, semantic_description: str) -> str:
        """
        Create a detailed prompt for inpainting from semantic description.
        
        Args:
            semantic_description: Description like "Environment: urban setting 3 object(s) detected."
            
        Returns:
            Detailed prompt for inpainting model
        """
        # Parse environment type from description
        # Format: "Environment: <category> N object(s) detected."
        
        prompt_mapping = {
            "indoor": "indoor environment, interior room, furniture",
            "outdoor": "outdoor environment, nature, landscape",
            "urban": "urban setting, city street, buildings, architecture",
            "natural": "natural landscape, nature, trees, grass",
            "bright": "bright and well-lit, sunny, good lighting",
            "dark": "dark or dimly lit, night scene, shadows",
            "daytime": "daytime scene, sunny, bright natural light",
            "nighttime": "nighttime scene, dark, artificial lighting",
            "close-up": "close-up view, detailed foreground",
            "wide": "wide landscape view, panoramic, expansive",
            "minimalist": "minimalist background, clean, simple",
            "complex": "complex detailed background, rich details",
        }
        
        # Extract keywords from description
        description_lower = semantic_description.lower()
        matched_prompts = []
        
        for keyword, prompt in prompt_mapping.items():
            if keyword in description_lower:
                matched_prompts.append(prompt)
        
        # Build final prompt
        if matched_prompts:
            base_prompt = ", ".join(matched_prompts[:3])  # Use top 3 matches
        else:
            base_prompt = "realistic scene with natural composition"
        
        # Add general quality modifiers
        final_prompt = f"{base_prompt}, high quality, realistic, well composed, 8k"
        
        return final_prompt

