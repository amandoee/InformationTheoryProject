"""Test the image receiver reconstruction pipeline."""

from receiver import ImageReceiver
import os

if __name__ == "__main__":
    print("=" * 70)
    print("IMAGE RECEIVER - RECONSTRUCTION PIPELINE TEST")
    print("=" * 70)
    print()
    
    # Test 1: Reconstruct with subimages only (fast, no model needed)
    print("Step 1: Load Receiver (with lazy model loading)")
    print("-" * 70)
    try:
        receiver = ImageReceiver(lazy_load_model=True)
        print("✓ Receiver initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing receiver: {e}")
        exit(1)
    
    print()
    print("Step 2: Reconstruct with subimages only (no background generation)")
    print("-" * 70)
    print("This creates a partial image with only the detected objects")
    print("placed at their original positions, rest is black.")
    print()
    try:
        partial = receiver.reconstruct_from_subimages(
            subimages_dir="output",
            semantic_json_path="output/result_semantic.json",
            output_path="output/result_partial.png",
            generate_background=False
        )
        print(f"✓ Partial reconstruction successful!")
        print(f"  Output saved to: output/result_partial.png")
        print(f"  Shape: {partial.shape}")
        
        # Check that file exists
        if os.path.exists("output/result_partial.png"):
            file_size = os.path.getsize("output/result_partial.png")
            print(f"  File size: {file_size} bytes")
    except Exception as e:
        print(f"✗ Error during partial reconstruction: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("Step 3: Reconstruct with AI background generation")
    print("-" * 70)
    print("This uses Stable Diffusion to inpaint the missing areas based on")
    print("the semantic description (e.g., 'urban setting')")
    print()
    print("NOTE: First run will download Stable Diffusion model (~5.5GB)")
    print("      Subsequent runs will use cached model")
    print()
    
    try:
        response = input("Proceed with AI background generation? (y/n): ").strip().lower()
        if response == 'y':
            full = receiver.reconstruct_from_subimages(
                subimages_dir="output",
                semantic_json_path="output/result_semantic.json",
                output_path="output/result_reconstructed.png",
                generate_background=True,
                inference_steps=15
            )
            print()
            print(f"✓ Full reconstruction successful!")
            print(f"  Output saved to: output/result_reconstructed.png")
            print(f"  Shape: {full.shape}")
            
            if os.path.exists("output/result_reconstructed.png"):
                file_size = os.path.getsize("output/result_reconstructed.png")
                print(f"  File size: {file_size} bytes")
        else:
            print("Skipped AI background generation")
    except KeyboardInterrupt:
        print()
        print("Cancelled by user")
    except Exception as e:
        print(f"✗ Error during full reconstruction: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("RECEIVER TEST COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("--------")
    print("✓ Sender creates: subimages + semantic description")
    print("✓ Receiver reconstructs: objects placed at original positions")
    print("✓ Optional: Receiver uses AI to inpaint missing background areas")
    print()

