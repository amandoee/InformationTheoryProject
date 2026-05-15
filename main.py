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
    encoder.encode(output_file)

    original_size = os.path.getsize(image_path) / 1024
    compressed_size = os.path.getsize("test") / 1024
    print(f"COMPRESSION: {original_size:.1f} KB → {compressed_size:.1f} KB")

    return True

def decompress(asset_path: str, output_file: str):
    
    #implement decompressor to read compressed assets
    #and write the assets in the desired way to the temp folder

    #TODO: make decompressor do all writes in temp folder, just like manøes pipeline
    #TODO: generate the json file in the desired format, such that it can be used by the pipeline
    output_prefix = "temp"

    # Verify transmitted data exists
    semantic_json_path = f"{output_prefix}_semantic.json"
    if not os.path.exists(semantic_json_path):
        print(f"✗ {semantic_json_path} NOT FOUND")
        return False
    
    # Dynamically find all subimages
    subimage_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("subimage_") and f.endswith(".png")
    ])

    try:
        receiver = ImageReceiver(lazy_load_model=True)
        print("  ✓ Ready (Stable Diffusion lazy-loaded)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    try:
        receiver.reconstruct_from_subimages(
            subimages_dir=output_dir,
            semantic_json_path=semantic_json_path,
            output_path=os.path.join(output_dir, "e2e_partial.png"),
            generate_background=False
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        full = receiver.reconstruct_from_subimages(
            subimages_dir=output_dir,
            semantic_json_path=semantic_json_path,
            output_path=os.path.join(output_dir, output_file),
            generate_background=True,
            inference_steps=30
        )
    except Exception as e:
        print(f"  ✗ Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True




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
    