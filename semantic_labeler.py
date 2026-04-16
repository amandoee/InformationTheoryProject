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


def segment_image(image : np.ndarray, bounding_boxes) -> SegmentedData:
        
        image_height, image_width, _ = image.shape
        subimage_with_coordinate_and_class=[]
        #extract each image
        for box in bounding_boxes:

            class_id, x_center, y_center, width, height = box

            
            (top_y, top_x) = int(np.round(np.min(y_center-height/2)), int(np.round(np.min(x_center-width/2))))
            (bottom_y, bottom_x) = (np.max(y_center+height/2), np.max(x_center+width/2))

            # Crop image (extract)
            extracted_image = image[top_y:bottom_y+1, top_x:bottom_x+1]


            new_subimage = subImage(imagedata=extracted_image,
                                    x_upperleft_coordinate=x_center-(width/2),
                                    y_upperleft_coordinate=y_center-(height/2),
                                    class_id=class_id)

            subimage_with_coordinate_and_class.append(new_subimage)


        segmented_data = SegmentedData(image_height,image_width, subimage_with_coordinate_and_class)   

        return segmented_data