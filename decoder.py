from construct import Struct, Int32ub, PascalString, PrefixedArray, Bytes
from semantic_labeler import subImage
import struct
import numpy as np
import cv2
import os
import json


class Decoder:
    def __init__(self, asset_path: str):
        self.asset_path = asset_path
        self.global_height = 0
        self.global_width = 0
        self.num_samples = 0
    """
    read contents from compressed assets and write
    necesarry files to temp_dir
    """
    def _read_contents (self, temp_dir):
      header_size = 3 * 4
      bbox_list = []

      with open(self.asset_path,'rb') as asset_file:
          header_data = asset_file.read(header_size)
          self.global_height, self.global_width, self.num_samples = struct.unpack('<III', header_data)

          description_size_bytes = asset_file.read(4)

          description_size = struct.unpack('<I', description_size_bytes)[0]

          description = asset_file.read(description_size).decode('utf-8')

          for i in range(self.num_samples):
              size_data = asset_file.read(4)
              size_of_jpeg = struct.unpack('<I', size_data)[0]

              jpeg_bytes = asset_file.read(size_of_jpeg)
              jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
              imagedata = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

              bbox_bytes = asset_file.read(12)
              x, y, class_id = struct.unpack('<iii', bbox_bytes)
              bbox_list.append({
                  "class_id": class_id,
                  "x_position": x,
                  "y_position": y,
                  "height": imagedata.shape[0],  # Rows
                  "width": imagedata.shape[1]    # Columns
              })

              output_img_path = os.path.join(temp_dir, f"subimage_{i}.png")
              cv2.imwrite(output_img_path, imagedata)

      self._write_json(temp_dir, description, bbox_list)

    def _write_json(self, temp_dir: str, description: str, bboxes: list):
        """
        Saves the structured JSON metadata file into the temp directory.
        """
        json_data = {
            "original_height": self.global_height,
            "original_width": self.global_width,
            "semantic_description": description,
            "num_subimages": self.num_samples,
            "subimages": bboxes
        }

        json_output_path = os.path.join(temp_dir, "semantic.json")
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2)
            
        print(f"Metadata file successfully saved to: {json_output_path}")
    
    def decompress(self, temp_dir):
        
        os.makedirs(temp_dir, exist_ok=True)
        self._read_contents(temp_dir)
