# DINOv3 Feature Similarity Integration

## What it does

Adds a "DINO-Assisted" annotation tool that finds visually similar regions across an image using DINOv3 patch features and cosine similarity.

**Workflow:**
1. Select a class and click "DINO-Assisted" in the sidebar
2. Draw a bounding box around a region of interest
3. The tool extracts DINOv3 patch features from inside the box, computes cosine similarity against all patches in the image, and highlights matching regions as proposed annotations (orange dashed outlines)
4. Adjust the threshold slider and release to recompute with the new threshold
5. Press **Enter** to accept the proposed annotations, or **Escape** to discard them

## Technical details

- **Model**: `dinov3_vitl16` (ViT-Large, 1024-dim features, 24 layers)
- **Resolution**: Images are downscaled to 2048px height (width proportional, both divisible by 16) for feature extraction
- **Patch size**: 16x16 pixels — each patch gets a 1024-dim feature vector
- **Similarity**: Centroid aggregation (average features inside bbox) + cosine similarity against all patches
- **Minimum region size**: Connected components smaller than 10 patches are discarded
- **Caching**: Features are extracted once per image and cached — subsequent bbox queries on the same image are instant
- **Device**: CUDA with automatic CPU fallback

## Dependencies

- DINOv3 package at `/home/xa/PCS/tree_counter_dinov3/tree_counter/dinov3/`
- Model weights at `/home/xa/PCS/tree_counter_dinov3/tree_counter_clean/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

## Files changed

- **`src/digitalsreeni_image_annotator/dino_utils.py`** (new) — DINOUtils class: model loading, feature extraction, cosine similarity, patch-to-polygon conversion
- **`src/digitalsreeni_image_annotator/image_label.py`** — Extended SAM bbox drawing conditions to also work for DINO tool
- **`src/digitalsreeni_image_annotator/annotator_window.py`** — DINO button, threshold slider, activate/deactivate/apply methods, threshold recompute on slider release
