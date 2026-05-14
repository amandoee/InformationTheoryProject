# Kilde til inspiration med cutting a billder: https://stackoverflow.com/a/64376764

import numpy as np
from dataclasses import dataclass

@dataclass
class subImage:
        imagedata : np.ndarray
        x_upperleft_coordinate : int
        y_upperleft_coordinate : int
        class_id : int
        
@dataclass
class SegmentedData:
        original_height : int
        original_width : int
        list_of_subimages : list[subImage]
        semantic_description : str = ""


def segment_image(image : np.ndarray, bounding_boxes, normalized: bool = True) -> SegmentedData:

        
        image_height, image_width, _ = image.shape
        subimage_with_coordinate_and_class = []
        
        # Extract each image
        for box in bounding_boxes:
            class_id, x_center, y_center, width, height = box
            
            # Convert normalized coordinates to pixel coordinates if needed
            if normalized:
                x_center = x_center * image_width
                y_center = y_center * image_height
                width = width * image_width
                height = height * image_height
            
            # Calculate corners
            top_x = max(0, int(np.round(x_center - width/2)))
            top_y = max(0, int(np.round(y_center - height/2)))
            bottom_x = min(image_width, int(np.round(x_center + width/2)))
            bottom_y = min(image_height, int(np.round(y_center + height/2)))

            # Crop image (extract)
            extracted_image = image[top_y:bottom_y, top_x:bottom_x]
            
            if extracted_image.size == 0:  # Skip empty crops
                continue

            new_subimage = subImage(imagedata=extracted_image,
                                    x_upperleft_coordinate=top_x,
                                    y_upperleft_coordinate=top_y,
                                    class_id=class_id)

            subimage_with_coordinate_and_class.append(new_subimage)

        segmented_data = SegmentedData(image_height, image_width, subimage_with_coordinate_and_class)   

        return segmented_data


def reconstruct_subimages_only(segmented_data: SegmentedData) -> np.ndarray:
        """
        Create a new image containing only the extracted subimages placed at their
        original positions in the image. Everything else is cut out (filled with black).
        
        Args:
            segmented_data: SegmentedData object containing the subimages and their positions
            
        Returns:
            Image of original size with only subimages placed, rest is black
        """
        # Create black canvas of original size
        reconstructed = np.zeros((segmented_data.original_height, 
                                  segmented_data.original_width, 3), dtype=np.uint8)
        
        # Place each subimage at its original position
        for subimage in segmented_data.list_of_subimages:
                top_x = int(subimage.x_upperleft_coordinate)
                top_y = int(subimage.y_upperleft_coordinate)
                height, width = subimage.imagedata.shape[:2]
                
                # Ensure we don't go out of bounds
                bottom_x = min(top_x + width, segmented_data.original_width)
                bottom_y = min(top_y + height, segmented_data.original_height)
                
                # Handle case where subimage might have been cropped at edges
                img_height = bottom_y - top_y
                img_width = bottom_x - top_x
                
                # Place subimage data in canvas
                reconstructed[top_y:bottom_y, top_x:bottom_x] = subimage.imagedata[:img_height, :img_width]
        
        return reconstructed


def create_semantic_dict(segmented_data: SegmentedData) -> dict:
        """
        Create a dictionary representation of the segmented data for semantic storage.
        This can be used to reconstruct the image even without the actual image data.
        
        Args:
            segmented_data: SegmentedData object
            
        Returns:
            Dictionary containing all metadata needed for reconstruction
        """
        subimages_info = []
        for subimage in segmented_data.list_of_subimages:
                subimages_info.append({
                        'class_id': int(subimage.class_id),
                        'x_position': int(subimage.x_upperleft_coordinate),
                        'y_position': int(subimage.y_upperleft_coordinate),
                        'height': subimage.imagedata.shape[0],
                        'width': subimage.imagedata.shape[1],
                })
        
        return {
                'original_height': segmented_data.original_height,
                'original_width': segmented_data.original_width,
                'semantic_description': segmented_data.semantic_description,
                'num_subimages': len(segmented_data.list_of_subimages),
                'subimages': subimages_info
        }


def print_semantic_info(segmented_data: SegmentedData):
        """
        Print semantic information about the segmented data.
        
        Args:
            segmented_data: SegmentedData object
        """
        print("=" * 60)
        print("SEMANTIC IMAGE INFORMATION")
        print("=" * 60)
        print(f"Original Image Size: {segmented_data.original_width}x{segmented_data.original_height}")
        print(f"\nImage Description:\n{segmented_data.semantic_description}")
        print(f"\nExtracted Objects: {len(segmented_data.list_of_subimages)}")
        print("-" * 60)
        
        for idx, subimage in enumerate(segmented_data.list_of_subimages):
                height, width = subimage.imagedata.shape[:2]
                print(f"Object {idx + 1}:")
                print(f"  Class ID: {subimage.class_id}")
                print(f"  Position: ({subimage.x_upperleft_coordinate}, {subimage.y_upperleft_coordinate})")
                print(f"  Size: {width}x{height} pixels")
                print(f"  Coverage: {(width * height) / (segmented_data.original_width * segmented_data.original_height) * 100:.2f}%")