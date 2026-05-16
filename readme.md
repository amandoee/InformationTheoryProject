# Semantic Image Compression Pipeline
Our Information Theory project that extracts semantically important image snippets, packs them into a custom binary file, and reconstructs the image using generative background inpainting.

# Usage

| Short Flag | Long Flag | Description |
| :--- | :--- | :--- |
| `-c` | `--Compress` | Path to the original input image file to be compressed. |
| `-d` | `--Decompress` | Path to the custom compressed binary asset to be decoded. |
| `-o` | `--Output` | Destination path for the final generated output file. |

## Compressor

python main.py -c sample.jpg -o compressed_asset.bin

## Decompressor

python main.py -d compressed_asset.bin -o reconstructed_output.png