"""Main pipeline connecting detection, segmentation, and visualization."""

import cv2
from typing import List, Tuple
from detect import ObjectDetector, load_image
from semantic_labeler import segment_image, reconstruct_subimages_only, print_semantic_info, create_semantic_dict
from semantic_description import SemanticDescriber, generate_description_with_detections
from visualizer import ImageVisualizer, visualize_detection_and_segmentation


class DetectionPipeline:
    
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5):

        self.detector = ObjectDetector(conf_threshold=conf_threshold)
        self.model_name = model_name
        self.semantic_describer = SemanticDescriber()
    
    def run(self, image_path: str, output_prefix: str = None, 
            add_semantic_description: bool = True) -> Tuple:
        
        image = load_image(image_path)
        
        bounding_boxes = self.detector.detect_with_ultralytics(image, self.model_name)
        
        if not bounding_boxes:
            return image, bounding_boxes, None
        
        segmented_data = segment_image(image, bounding_boxes)
        
        # Add semantic description
        if add_semantic_description:
            segmented_data.semantic_description = generate_description_with_detections(
                image, bounding_boxes
            )
        
        if output_prefix:
            self._save_results(image, bounding_boxes, segmented_data, output_prefix)
        
        return image, bounding_boxes, segmented_data
    
    def _save_results(self, image, bounding_boxes, segmented_data, output_prefix: str):
        """Save all visualizations and results."""
        import os        
        # Create output directory if needed
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save bounding boxes visualization
        viz_boxes = ImageVisualizer.draw_bounding_boxes(image, bounding_boxes)
        cv2.imwrite(f"{output_prefix}_boxes.png", viz_boxes)
        
        # Save segmented regions visualization
        viz_regions = ImageVisualizer.create_segmentation_visualization(image, segmented_data)
        cv2.imwrite(f"{output_prefix}_regions.png", viz_regions)
        
        # Save subimages-only reconstruction (everything else cut out)
        subimages_only = reconstruct_subimages_only(segmented_data)
        cv2.imwrite(f"{output_prefix}_subimages_only.png", subimages_only)
        print(f"  Saved: {output_prefix}_subimages_only.png")
        
        # Save subimages grid
        viz_grid = ImageVisualizer.create_subimage_grid(segmented_data)
        cv2.imwrite(f"{output_prefix}_grid.png", viz_grid)
        
        # Save combined visualization
        combined = ImageVisualizer.create_combined_visualization(image, bounding_boxes, segmented_data)
        cv2.imwrite(f"{output_prefix}_combined.png", combined)
        
        # Save individual subimages for receiver
        self._save_subimages(segmented_data, output_prefix)
        
        # Save semantic information to file
        self._save_semantic_info(segmented_data, output_prefix)
        
        print(f"Results saved successfully!")
    
    def _save_subimages(self, segmented_data, output_prefix: str):
        """Save individual subimages to separate files for the receiver."""
        import os
        
        output_dir = os.path.dirname(output_prefix) or "."
        
        for i, subimage in enumerate(segmented_data.list_of_subimages):
            subimage_path = os.path.join(output_dir, f"subimage_{i}.png")
            cv2.imwrite(subimage_path, subimage.imagedata)
    
    def _save_semantic_info(self, segmented_data, output_prefix: str):
        """Save semantic information to JSON and text files."""
        import json
        
        # Print to console
        print_semantic_info(segmented_data)
        
        # Save to JSON file
        semantic_dict = create_semantic_dict(segmented_data)
        json_path = f"{output_prefix}_semantic.json"
        with open(json_path, 'w') as f:
            json.dump(semantic_dict, f, indent=2)
        
        # Save description to text file
        text_path = f"{output_prefix}_description.txt"
        with open(text_path, 'w') as f:
            f.write("IMAGE SEMANTIC DESCRIPTION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Original Image Size: {segmented_data.original_width}x{segmented_data.original_height}\n\n")
            f.write("Description:\n")
            f.write(segmented_data.semantic_description + "\n\n")
            f.write(f"Total Objects Extracted: {len(segmented_data.list_of_subimages)}\n")


def run_pipeline_simple(image_path: str, model_name: str = "yolov8n.pt", 
                       output_prefix: str = None):

    pipeline = DetectionPipeline(model_name=model_name)
    image, boxes, segmented = pipeline.run(image_path, output_prefix)
    return image, boxes, segmented

