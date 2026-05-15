"""
Complete end-to-end pipeline: SENDER → RECEIVER

Demonstrates the full information-theoretic pipeline:
1. SENDER: Detect objects → Extract subimages → Generate semantic description
2. RECEIVER: Receive artifacts → Reconstruct image → Inpaint background
"""

import os
import sys
from pipeline import run_pipeline_simple
from receiver import ImageReceiver

def main(image_path: str = "sample.jpg", output_dir: str = "output"):
    """
    Run complete end-to-end pipeline.
    
    Args:
        image_path: Path to input image (default: sample.jpg)
        output_dir: Directory for outputs (default: output)
    """
    
    # Validate input image
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False
    
    print("=" * 80)
    print("COMPLETE END-TO-END PIPELINE: SENDER -> RECEIVER")
    print("=" * 80)
    print()
    
    print("SENDER SIDE")
    print("-" * 80)
    
    output_prefix = os.path.join(output_dir, "result")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run sender pipeline (CLIP-only, YOLOv8 detection + CLIP description)
        image, bounding_boxes, segmented_data = run_pipeline_simple(
            image_path,
            output_prefix=output_prefix
        )
    except Exception as e:
        print(f"✗ Sender pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Sender complete")
    print()
    
    # Display what was sent
    num_objects = len(segmented_data.list_of_subimages)
    print(f"TRANSMITTED DATA:")
    print(f"  ├─ Description: {segmented_data.semantic_description}")
    print(f"  ├─ Objects: {num_objects}")
    print(f"  └─ Canvas: {segmented_data.original_height}×{segmented_data.original_width}")
    print()
    
    # Calculate compression
    original_size = os.path.getsize(image_path) / 1024
    transmitted_data = sum(
        os.path.getsize(os.path.join(output_dir, f"subimage_{i}.png")) / 1024
        for i in range(num_objects)
        if os.path.exists(os.path.join(output_dir, f"subimage_{i}.png"))
    ) + os.path.getsize(f"{output_prefix}_semantic.json") / 1024
    
    compression_ratio = original_size / transmitted_data
    
    print(f"COMPRESSION: {original_size:.1f} KB → {transmitted_data:.1f} KB ({compression_ratio:.1f}:1)")
    print()
    
    print("RECEIVER SIDE")
    print("-" * 80)
    
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
    
    if not subimage_files:
        print(f"✗ No subimage files found in {output_dir}")
        return False
    
    print(f"✓ Received metadata and {len(subimage_files)} subimages")
    print()
    
    # Initialize receiver
    print("Step 1: Initialize Receiver")
    try:
        receiver = ImageReceiver(lazy_load_model=True)
        print("  ✓ Ready (Stable Diffusion lazy-loaded)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print()
    



    
    # Phase 1: Partial reconstruction
    print("Step 2: Partial Reconstruction (objects only)")
    try:
        partial = receiver.reconstruct_from_subimages(
            subimages_dir=output_dir,
            semantic_json_path=semantic_json_path,
            output_path=os.path.join(output_dir, "e2e_partial.png"),
            generate_background=False
        )
        print(f"  ✓ Complete ({partial.shape[1]}×{partial.shape[0]}px)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Phase 2: Full reconstruction
    print("Step 3: Full Reconstruction (with AI inpainting)")
    try:
        full = receiver.reconstruct_from_subimages(
            subimages_dir=output_dir,
            semantic_json_path=semantic_json_path,
            output_path=os.path.join(output_dir, "e2e_full_reconstructed.png"),
            generate_background=True,
            inference_steps=30
        )
        print(f"  ✓ Complete ({full.shape[1]}×{full.shape[0]}px)")
    except Exception as e:
        print(f"  ✗ Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 80)
    print("SUCCESS: PIPELINE COMPLETE")
    print("=" * 80)
    print()
    print("SUMMARY:")
    print(f"  Input: {image_path} ({original_size:.1f} KB)")
    print(f"  Objects detected: {num_objects}")
    print(f"  Compression: {compression_ratio:.1f}:1")
    print()
    print(f"OUTPUT FILES (in {output_dir}/):")
    print(f"  ├─ e2e_partial.png - Objects only")
    print(f"  └─ e2e_full_reconstructed.png - With AI background")
    print()
    
    return True

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample2.jpg"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "output"
    
    success = main(image_path, output_dir)
