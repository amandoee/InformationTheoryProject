from construct import Struct, Int32ub, PascalString, PrefixedArray, Bytes
from semantic_labeler import subImage
import struct
import numpy as np
import cv2

class Encoder:
    """
    Encodes semantically important subimages(Snippets) S_i in jpeg format
    We also encode relevant metadata for lossy image reconstruction of background
    The encoding format will be something like this:

    height     | width | Number of snippets |
    SizeOf S_0 | S_0   | bbox_0 | SizeOf S_1 | S_1 | bbox_1
    |             ...           | SizeOf S_N | S_N | bbox_N
    """
    def __init__(self, subimages: list[subImage], bbox, description, imageDimensions):
        """
        entries: List of tuples [(raw image), (x, y, width, height)]
        """
        self.subimages = subimages
        self.snippets = len(subimages)
        self.bbox = bbox
        self.height, self.width = imageDimensions
        self.quality = 80
        self.description = description
    def encode(self, outfile):

        with open(outfile,'wb') as outfile:
            header = struct.pack('<III', self.height, self.width,self.snippets)
            outfile.write(header)

            description_bytes = self.description.encode('utf-8')

            outfile.write(struct.pack('<I',len(description_bytes)))
            print(self.description)
            print("string length: ",len(description_bytes))
            outfile.write(description_bytes)

            for subimage in self.subimages:
                jpeg_bytes = self._jpeg_encoder(subimage.imagedata)
                size_of_s = len(jpeg_bytes)

                outfile.write(struct.pack('<I',size_of_s))
                outfile.write(jpeg_bytes)
                bbox_data = struct.pack('<III', subimage.x_upperleft_coordinate,subimage.y_upperleft_coordinate,subimage.class_id)
                outfile.write(bbox_data)
    def _jpeg_encoder(self, image_data: np.ndarray) -> bytes:
        """s
        Encodes a single numpy array into JPEG bytes.
        """
        # Encode parameters: [cv2.IMWRITE_JPEG_QUALITY, value]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        
        success, encoded_img = cv2.imencode('.jpg', image_data, encode_param)
        if not success:
            raise ValueError("JPEG encoding failed")
            
        return encoded_img.tobytes()
