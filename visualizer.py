"""Visualization module for displaying detected regions and extracted subimages."""

import cv2
import numpy as np
from typing import List, Tuple
from semantic_labeler import SegmentedData, subImage, reconstruct_subimages_only


class ImageVisualizer:
    """Visualize detection results and segmented regions."""
    
    @staticmethod
    def draw_bounding_boxes(image: np.ndarray, bounding_boxes: List[Tuple], 
                           color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image (BGR format)
            bounding_boxes: List of boxes in format (class_id, x_center, y_center, width, height)
                           where coordinates are normalized to [0, 1]
            color: Box color in BGR format
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        height, width, _ = image.shape
        
        for box in bounding_boxes:
            class_id, x_center, y_center, box_width, box_height = box
            
            # Convert normalized coordinates to pixel coordinates
            x_center_px = int(x_center * width)
            y_center_px = int(y_center * height)
            width_px = int(box_width * width)
            height_px = int(box_height * height)
            
            # Calculate top-left and bottom-right corners
            top_left_x = max(0, int(x_center_px - width_px / 2))
            top_left_y = max(0, int(y_center_px - height_px / 2))
            bottom_right_x = min(width, int(x_center_px + width_px / 2))
            bottom_right_y = min(height, int(y_center_px + height_px / 2))
            
            # Draw rectangle
            cv2.rectangle(result, (top_left_x, top_left_y), 
                         (bottom_right_x, bottom_right_y), color, thickness)
            
            # Draw class ID
            cv2.putText(result, f"Class {class_id}", (top_left_x, top_left_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    @staticmethod
    def create_segmentation_visualization(original_image: np.ndarray, 
                                         segmented_data: SegmentedData,
                                         show_grid: bool = True) -> np.ndarray:
        """
        Create visualization showing segmented regions on original image.
        
        Args:
            original_image: Original image
            segmented_data: SegmentedData object containing subimages and coordinates
            show_grid: Whether to show grid lines for extracted regions
            
        Returns:
            Image showing all extracted regions marked on original size canvas
        """
        result = original_image.copy()
        
        # Color palette for different regions
        colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
        ]
        
        for idx, subimage in enumerate(segmented_data.list_of_subimages):
            color = colors[idx % len(colors)]
            
            top_x = int(subimage.x_upperleft_coordinate)
            top_y = int(subimage.y_upperleft_coordinate)
            height, width = subimage.imagedata.shape[:2]
            
            # Clamp coordinates to image bounds
            top_x = max(0, top_x)
            top_y = max(0, top_y)
            bottom_x = min(segmented_data.original_width, top_x + width)
            bottom_y = min(segmented_data.original_height, top_y + height)
            
            # Draw rectangle
            cv2.rectangle(result, (top_x, top_y), (bottom_x, bottom_y), 
                         color, 2)
            
            # Draw region label
            label = f"Region {idx}\nClass {subimage.class_id}"
            cv2.putText(result, label, (top_x + 5, top_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result
    
    @staticmethod
    def create_subimage_grid(segmented_data: SegmentedData, 
                            cols: int = 3, 
                            cell_width: int = 150) -> np.ndarray:
        """
        Create a grid visualization of all extracted subimages.
        
        Args:
            segmented_data: SegmentedData object containing subimages
            cols: Number of columns in the grid
            cell_width: Width of each cell in the grid
            
        Returns:
            Grid image showing all subimages
        """
        subimages = segmented_data.list_of_subimages
        if not subimages:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        num_images = len(subimages)
        rows = (num_images + cols - 1) // cols
        
        # Calculate dimensions
        cell_height = cell_width
        grid_width = cols * (cell_width + 10) + 10
        grid_height = rows * (cell_height + 10) + 10
        
        # Create white background
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        for idx, subimage in enumerate(subimages):
            row = idx // cols
            col = idx % cols
            
            x_start = col * (cell_width + 10) + 10
            y_start = row * (cell_height + 10) + 10
            
            # Resize subimage to fit cell
            resized = cv2.resize(subimage.imagedata, (cell_width, cell_height))
            
            # Place in grid
            grid[y_start:y_start + cell_height, x_start:x_start + cell_width] = resized
            
            # Draw border
            cv2.rectangle(grid, (x_start, y_start), 
                         (x_start + cell_width, y_start + cell_height),
                         (0, 0, 0), 1)
            
            # Add label
            label = f"Cls {subimage.class_id}"
            cv2.putText(grid, label, (x_start + 5, y_start + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return grid
    
    @staticmethod
    def create_combined_visualization(original_image: np.ndarray,
                                     bounding_boxes: List[Tuple],
                                     segmented_data: SegmentedData) -> np.ndarray:
        """
        Create a combined visualization showing:
        1. Original image with bounding boxes
        2. Segmented regions on original image
        3. Reconstructed image (only subimages)
        4. Grid of all extracted subimages
        
        Args:
            original_image: Original image
            bounding_boxes: List of detected bounding boxes
            segmented_data: Segmented data from semantic_labeler
            
        Returns:
            Combined visualization image
        """
        # Create individual visualizations
        with_boxes = ImageVisualizer.draw_bounding_boxes(original_image, bounding_boxes)
        with_regions = ImageVisualizer.create_segmentation_visualization(original_image, segmented_data)
        reconstructed = reconstruct_subimages_only(segmented_data)
        subimage_grid = ImageVisualizer.create_subimage_grid(segmented_data)
        
        # Calculate target height for bottom row (half of top row)
        top_row = np.hstack([with_boxes, with_regions])
        target_height = top_row.shape[0] // 2
        
        # Resize reconstructed and grid to match target height
        if reconstructed.shape[0] != target_height:
            reconstructed = cv2.resize(reconstructed, 
                                      (int(reconstructed.shape[1] * target_height / reconstructed.shape[0]), 
                                       target_height))
        
        if subimage_grid.shape[0] != target_height:
            subimage_grid = cv2.resize(subimage_grid, 
                                      (int(subimage_grid.shape[1] * target_height / subimage_grid.shape[0]), 
                                       target_height))
        
        # Stack bottom row
        bottom_row = np.hstack([reconstructed, subimage_grid])
        
        # Ensure rows have compatible width by padding if necessary
        if top_row.shape[1] > bottom_row.shape[1]:
            padding = top_row.shape[1] - bottom_row.shape[1]
            bottom_row = np.hstack([bottom_row, np.zeros((target_height, padding, 3), dtype=np.uint8)])
        elif bottom_row.shape[1] > top_row.shape[1]:
            padding = bottom_row.shape[1] - top_row.shape[1]
            top_row = np.hstack([top_row, np.zeros((top_row.shape[0], padding, 3), dtype=np.uint8)])
        
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    @staticmethod
    def save_visualization(image: np.ndarray, output_path: str):
        """
        Save visualization to file.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")


def visualize_detection_and_segmentation(image_path: str, 
                                        bounding_boxes: List[Tuple],
                                        segmented_data: SegmentedData,
                                        output_path: str = None):
    """
    Convenience function to visualize detection and segmentation results.
    
    Args:
        image_path: Path to original image
        bounding_boxes: List of detected bounding boxes
        segmented_data: Segmented data from semantic_labeler
        output_path: Optional path to save visualization
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create separate visualizations
    viz_boxes = ImageVisualizer.draw_bounding_boxes(image, bounding_boxes)
    viz_regions = ImageVisualizer.create_segmentation_visualization(image, segmented_data)
    viz_grid = ImageVisualizer.create_subimage_grid(segmented_data)
    
    # Save individual visualizations if output path provided
    if output_path:
        base_path = output_path.rsplit('.', 1)[0]
        cv2.imwrite(f"{base_path}_boxes.png", viz_boxes)
        cv2.imwrite(f"{base_path}_regions.png", viz_regions)
        cv2.imwrite(f"{base_path}_grid.png", viz_grid)
        print(f"Visualizations saved with prefix: {base_path}")
    
    return viz_boxes, viz_regions, viz_grid
