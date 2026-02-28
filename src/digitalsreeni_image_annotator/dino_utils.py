"""
DINOv3 feature extraction and cosine similarity for finding similar patches.
Uses dinov3_vitl16 (1024-dim, 24 layers) at 2048 downscale resolution.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import cv2

# DINOv3 package and weights paths
_DINOV3_REPO = Path('/home/xa/PCS/tree_counter_dinov3/tree_counter/dinov3')
_WEIGHTS_DIR = Path('/home/xa/PCS/tree_counter_dinov3/tree_counter_clean/weights')

PATCH_SIZE = 16
IMAGE_SIZE = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_LAYERS = 24
DIM = 1024


class DINOUtils:
    def __init__(self):
        self.model = None
        self.device = None
        self.feature_cache = {}  # image_path -> dict with features_norm, metadata

    def load_model(self):
        """Load dinov3_vitl16 from local weights. Returns True on success."""
        if str(_DINOV3_REPO) not in sys.path:
            sys.path.insert(0, str(_DINOV3_REPO))

        from dinov3.hub.backbones import dinov3_vitl16

        weights_path = _WEIGHTS_DIR / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        if not weights_path.exists():
            print(f"DINO weights not found: {weights_path}")
            return False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = dinov3_vitl16(pretrained=False)
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            self.model = model
            print(f"DINOv3 vitl16 loaded on {self.device.upper()}")
            return True
        except Exception as e:
            print(f"Failed to load DINOv3: {e}")
            return False

    def extract_features(self, image_path):
        """Extract and cache L2-normalized patch features for an image."""
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]

        if self.model is None:
            if not self.load_model():
                return None

        image = Image.open(image_path).convert('RGB')
        original_w, original_h = image.size

        # Resize to patch-aligned dimensions at IMAGE_SIZE height
        h_patches = IMAGE_SIZE // PATCH_SIZE
        w_patches = int((original_w * IMAGE_SIZE) / (original_h * PATCH_SIZE))
        resized_h = h_patches * PATCH_SIZE
        resized_w = w_patches * PATCH_SIZE

        image_tensor = TF.to_tensor(TF.resize(image, (resized_h, resized_w)))
        image_normalized = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with torch.inference_mode():
            image_batch = image_normalized.unsqueeze(0).to(self.device)
            feats = self.model.get_intermediate_layers(
                image_batch, n=range(N_LAYERS), reshape=False, norm=True
            )
            # get_intermediate_layers already strips CLS token internally
            patch_features = feats[-1].squeeze(0).cpu().numpy()  # (h_patches * w_patches, DIM)

        # L2 normalize
        norms = np.linalg.norm(patch_features, axis=1, keepdims=True)
        features_norm = patch_features / (norms + 1e-8)

        result = {
            'features_norm': features_norm,
            'h_patches': h_patches,
            'w_patches': w_patches,
            'original_size': (original_w, original_h),
            'resized_size': (resized_w, resized_h),
            'scale_x': original_w / resized_w,
            'scale_y': original_h / resized_h,
        }
        self.feature_cache[image_path] = result
        print(f"DINO features extracted: {h_patches}x{w_patches} patches, {features_norm.shape}")
        return result

    def find_similar_patches(self, image_path, bbox, threshold):
        """Find patches similar to those inside bbox using cosine similarity.

        Args:
            image_path: Path to the image file
            bbox: [x1, y1, x2, y2] in original image coordinates
            threshold: Cosine similarity threshold (0.0 to 1.0)

        Returns:
            List of polygons (each a flat list [x1,y1,x2,y2,...]) in original coords
        """
        cache = self.extract_features(image_path)
        if cache is None:
            return []

        features_norm = cache['features_norm']
        h_patches = cache['h_patches']
        w_patches = cache['w_patches']
        scale_x = cache['scale_x']
        scale_y = cache['scale_y']
        resized_w, resized_h = cache['resized_size']

        # Map bbox from original coords to patch grid
        x1, y1, x2, y2 = bbox
        px1 = max(0, int((x1 / scale_x) / PATCH_SIZE))
        py1 = max(0, int((y1 / scale_y) / PATCH_SIZE))
        px2 = min(w_patches - 1, int((x2 / scale_x) / PATCH_SIZE))
        py2 = min(h_patches - 1, int((y2 / scale_y) / PATCH_SIZE))

        # Collect query features from patches inside bbox
        query_indices = []
        for py in range(py1, py2 + 1):
            for px in range(px1, px2 + 1):
                query_indices.append(py * w_patches + px)

        if not query_indices:
            return []

        # Centroid aggregation
        query_vectors = features_norm[query_indices]
        query_vec = np.mean(query_vectors, axis=0, keepdims=True)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Cosine similarity
        similarities = np.dot(features_norm, query_vec.T).squeeze()
        similarities = np.clip(similarities, 0, 1)

        # Threshold to binary mask on patch grid
        sim_grid = similarities.reshape(h_patches, w_patches)
        binary_mask = (sim_grid >= threshold).astype(np.uint8)

        # Remove connected components with fewer than 10 patches
        num_labels, labels = cv2.connectedComponents(binary_mask)
        for label in range(1, num_labels):
            if np.sum(labels == label) < 10:
                binary_mask[labels == label] = 0

        # Upscale to resized pixel space
        pixel_mask = cv2.resize(binary_mask, (resized_w, resized_h),
                                interpolation=cv2.INTER_NEAREST)

        # Find contours
        contours, _ = cv2.findContours(pixel_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Scale contours to original image coords
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            scaled = contour.astype(np.float64)
            scaled[:, :, 0] *= scale_x
            scaled[:, :, 1] *= scale_y
            polygon = scaled.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)

        print(f"DINO: {len(polygons)} regions above threshold {threshold:.2f}")
        return polygons

    def clear_cache(self, image_path=None):
        """Clear feature cache for a specific image or all images."""
        if image_path:
            self.feature_cache.pop(image_path, None)
        else:
            self.feature_cache.clear()
