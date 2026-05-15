from construct import Struct, Int32ub, PascalString, PrefixedArray, Bytes
from semantic_labeler import subImage
import struct
import numpy as np
import cv2
import os



class Decoder:
    def __init__(self, asset_path: str):
        self.asset_path = asset_path

    """
    read contents from compressed assets and write
    necesarry files to temp_dir
    """
    def __read_contents__ (self, temp_dir):
        header_size = 3 * 8
        #get width and height from jpeg and x,y from encoding
        bbox_data = []

        #TODO:
        #write all images as png to temp with subimage_i names
        #create dictionary to write as json with all needed metadata
        #figure out which fields
        with open(self.asset_path,'r') as asset_file:
            header_data = asset_file.read(header_size)
            self.global_height,
            self.global_width,
            self.num_samples = struct.unpack('<III', header_data)
            description_size = asset_file.read(1)
            description = str(asset_file.read(description_size))

            for i in range(self.num_samples):
                size_data = asset_file.read(4)
                size_of_jpeg = struct.unpack('<I', size_data)[0]
                jpeg_bytes = asset_file.read(size_of_jpeg)
                jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                imagedata = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
                bbox_data = asset_file.read(12)
                x, y, class_id = struct.unpack('<iii', bbox_data)
                bbox_data.append((class_id, x, y, imagedata.shape[0],imagedata.shape[1]))

                cv2.imwrite(img = imagedata,filename=temp_dir+"/subimage_"+str(i))
        #write json with all needed data follow format in temp/res_samantic.json
        #this is can be copied from the pipeline code...

    """
    {
  "original_height": 683,
  "original_width": 1024,
  "semantic_description": "Environment: urban setting 3 object(s) detected.",
  "num_subimages": 3,
  "subimages": [
    {
      "class_id": 1,
      "x_position": 245,
      "y_position": 328,
      "height": 240,
      "width": 227
    },
    {
      "class_id": 2,
      "x_position": 386,
      "y_position": 244,
      "height": 153,
      "width": 296
    },
    {
      "class_id": 0,
      "x_position": 260,
      "y_position": 181,
      "height": 337,
      "width": 151
    }
  ]
}
    
    
    """
    def __write_json__(self,width,height,description,bboxes):
        pass    
    
    def decompress(self, temp_dir):
        output_dir = os.path.dirname(temp_dir)


        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.__read_contents__()
