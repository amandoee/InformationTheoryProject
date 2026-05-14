import os
import sys
import cv2

from pipeline import run_pipeline_simple
from receiver import ImageReceiver

from encoder import Encoder


def main(image_path: str, output_file: str):
    print("running compressor")

    if not os.path.exists(image_path):
        return -1
    
    intermediary_dir = "temp"
    
    temp_prefix = os.path.join(intermediary_dir,"res")
    os.makedirs(intermediary_dir,exist_ok=True)
    try:
        image, bounding_boxes, segmented_data = run_pipeline_simple(
            image_path,
            output_prefix=temp_prefix
        )
    except Exception as e:
        return -1
    encoder = Encoder(segmented_data.list_of_subimages,bounding_boxes,(image.shape[0],image.shape[1]))
    encoder.encode("test")
    original_size = os.path.getsize(image_path) / 1024
    compressed_size = os.path.getsize("test") / 1024
    print("Using our format")
    print(f"COMPRESSION: {original_size:.1f} KB → {compressed_size:.1f} KB")


    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        
    quality = 80
    _, encoded_img_standard = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    jpeg_size_kb = len(encoded_img_standard) / 1024

    print("Using JPEG format")

    print(f"COMPRESSION: {original_size:.1f} KB → {jpeg_size_kb:.1f} KB")






if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample.jpg"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "output"
    
    success = main(image_path, output_dir)