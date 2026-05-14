from pipeline import run_pipeline_simple

# Run pipeline with CLIP semantic descriptions
image, boxes, segmented = run_pipeline_simple(
    "sample.jpg",
    output_prefix="output/result"
)

# Access the semantic description
print(segmented.semantic_description)