import os
import getopt, sys
import cv2

from pipeline import run_pipeline_simple
from receiver import ImageReceiver

from encoder import Encoder

from enum import Enum

class state(Enum):
    COMPRESS = 1
    DECOMPRESS = 2

def compress(image_path: str, output_file: str):
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
        print(str(e))
        return -1
    encoder = Encoder(segmented_data.list_of_subimages,bounding_boxes,segmented_data.semantic_description,(image.shape[0],image.shape[1]))
    encoder.encode("test")
    original_size = os.path.getsize(image_path) / 1024
    compressed_size = os.path.getsize("test") / 1024
    print("Using our format")
    print(f"COMPRESSION: {original_size:.1f} KB → {compressed_size:.1f} KB")


    quality = 80
    _, encoded_img_standard = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    jpeg_size_kb = len(encoded_img_standard) / 1024

    print("Using JPEG format")

    print(f"COMPRESSION: {original_size:.1f} KB → {jpeg_size_kb:.1f} KB")

def decompress(asset_path: str, output_file: str):
    pass




if __name__ == "__main__":
    # Parse command-line arguments
    args = sys.argv[1:]
    options = "c:d:o:"
    long_options = ["Compress","Decompress","Output="]
    current_state = None
    output_dir = None
    asset_path = None

    try:
        arguements, values = getopt.getopt(args,options,long_options)
        for currentArg,current_val in arguements:
            if currentArg in ("-c", "--Compress"):
                current_state = state.COMPRESS
                asset_path = current_val
            elif currentArg in ("-d", "--Decompress"):
                current_state = state.DECOMPRESS
                asset_path = current_val
            elif currentArg in ("-o", "--Output"):
                output_dir = current_val
    except getopt.error as err:
        print(str(err))

    if(current_state == state.COMPRESS):
        success = compress(asset_path, output_dir)
    elif(current_state == state.DECOMPRESS):
        success = decompress(asset_path, output_dir)
    else:
        print("wah wah")
    