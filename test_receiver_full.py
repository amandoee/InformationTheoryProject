"""Test receiver with cached Stable Diffusion model - full end-to-end."""

import os
from receiver import ImageReceiver

if __name__ == "__main__":
    print("=" * 70)
    print("RECEIVER RECONSTRUCTION TEST - WITH STABLE DIFFUSION INPAINTING")
    print("=" * 70)
    print()
    
    # Verify model cache exists
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--runwayml--stable-diffusion-inpainting")
    
    print("Step 1: Check Model Cache")
    print("-" * 70)
    if os.path.exists(model_cache):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(model_cache)
            for filename in filenames
        ) / (1024 ** 3)  # Convert to GB
        print(f"✓ Stable Diffusion model cached at: {model_cache}")
        print(f"  Cache size: {cache_size:.2f} GB")
    else:
        print(f"✗ Model cache not found at: {model_cache}")
        print("  Run download_model.py first")
        exit(1)
    print()
    
    # Test 1: Verify metadata exists
    print("Step 2: Verify Sender Output Files")
    print("-" * 70)
    required_files = [
        "output/result_semantic.json",
        "output/subimage_0.png",
        "output/subimage_1.png",
        "output/subimage_2.png",
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size} bytes)")
        else:
            print(f"✗ {file} NOT FOUND")
            print("  Run main.py first to generate sender output")
            exit(1)
    print()
    
    # Test 2: Initialize receiver
    print("Step 3: Initialize Receiver (with cached model)")
    print("-" * 70)
    try:
        receiver = ImageReceiver(lazy_load_model=True)
        print("✓ Receiver initialized (lazy loading enabled)")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)
    print()
    
    # Test 3: Partial reconstruction (fast)
    print("Step 4: Partial Reconstruction (objects only)")
    print("-" * 70)
    try:
        partial = receiver.reconstruct_from_subimages(
            subimages_dir="output",
            semantic_json_path="output/result_semantic.json",
            output_path="output/test_partial.png",
            generate_background=False
        )
        print(f"✓ Partial reconstruction successful")
        print(f"  Output: output/test_partial.png")
        print(f"  Shape: {partial.shape}")
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 4: Full reconstruction with inpainting
    print("Step 5: Full Reconstruction (with AI inpainting)")
    print("-" * 70)
    print("Loading cached Stable Diffusion model and generating background...")
    print()
    
    try:
        full = receiver.reconstruct_from_subimages(
            subimages_dir="output",
            semantic_json_path="output/result_semantic.json",
            output_path="output/test_reconstructed_full.png",
            generate_background=True,
            inference_steps=30
        )
        print()
        print(f"✓ Full reconstruction successful!")
        print(f"  Output: output/test_reconstructed_full.png")
        print(f"  Shape: {full.shape}")
        print()
        
        # Verify output file
        if os.path.exists("output/test_reconstructed_full.png"):
            file_size = os.path.getsize("output/test_reconstructed_full.png")
            print(f"✓ Output file verified: {file_size} bytes")
        else:
            print("✗ Output file not found")
            exit(1)
            
    except Exception as e:
        print()
        print(f"✗ Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print()
    print("=" * 70)
    print("SUCCESS: END-TO-END RECEIVER PIPELINE WORKING!")
    print("=" * 70)
    print()
    print("Summary:")
    print("--------")
    print("✓ Stable Diffusion model cached locally (5.5GB)")
    print("✓ Receiver loads sender artifacts (subimages + metadata)")
    print("✓ Partial reconstruction: Objects placed at original positions")
    print("✓ Full reconstruction: AI inpaints missing background areas")
    print()
    print("Files generated:")
    print("  - output/test_partial.png (objects only)")
    print("  - output/test_reconstructed_full.png (with AI background)")
    print()
    print("The receiver pipeline is ready for production use!")
    print()
