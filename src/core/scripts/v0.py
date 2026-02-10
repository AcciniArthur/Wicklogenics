import os
import torch
import cv2
import sys

from src.core.scripts.modelPipeline import Model
from src.core.scripts.preprocessing.denoising import (
    denoise_image_tensor, load_and_preprocess_image, DenoisingModel,
    img_to_tensor, tensor_to_gray
)
from src.core.scripts.preprocessing.cropping import extract_tree_crop_from_image, CroppingModel
from src.core.scripts.preprocessing.labelling import extract_ordered_texts_from_image, LabellingModel, tensor_to_np
from src.core.scripts.nodesdetection.infer_ts_cpu import load_ts_model, infer_image
from src.core.scripts.postprocessing.corrector import correction


def resource_path(relative_path):
    """
    Retourne le bon chemin vers une ressource,
    en dev OU après build PyInstaller.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

MODELS_DIR = os.path.abspath(resource_path("../models"))

class v0(Model):
    def __init__(self):
        super().__init__()
        self._metadata["name"] = "V0"
        self._metadata["description"] = "Version 0"
        self._metadata["version"] = 0

        self.labelling_model = LabellingModel()
        self.cropping_model = CroppingModel()
        self.denoising_model = DenoisingModel()

        self.nodesdetection_model = None

    def load(self, device="cpu"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "..", "models")
        models_dir = os.path.abspath(models_dir)

        self.labelling_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "labelling_model.pth"), map_location=device))
        self.cropping_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "cropping_model.pth"), map_location=device))
        self.denoising_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "denoising_model.pth"), map_location=device))


        self.nodesdetection_model = load_ts_model(os.path.join(MODELS_DIR, "nodesdetection_model.pt"))

        self.labelling_model.eval()
        self.cropping_model.eval()
        self.denoising_model.eval()

        print("Models loaded")
        self._loaded = True

    def convert_points(self, image_path: str, device="cpu", conf_threshold: float = 0.5):
        """
        Pipeline:
        - load original image (H,W)
        - labelling on 512x512 (leaf names)
        - cropping on 512x512 -> returns crop + bbox in 512-space
        - map bbox to original image coordinates
        - crop original image using mapped bbox (important!)
        - denoise on cropped original
        - nodes detection on cropped original
        - correction
        - remap points back to original image (x += x0, y += y0)
        Return dict with nodes/corners/leaves + leaf_names + bbox, NO Newick.
        """
        self.ensure_loaded()

        # --- A) Load original (H,W) and also 512 version for labelling/cropping
        print("Loading image (original + 512)...")
        img_rgb_full, img_tensor_512, (H_full, W_full) = load_and_preprocess_image(image_path, size=(512, 512))
        # ATTENTION: load_and_preprocess_image te retourne img_rgb (probablement original) + tensor 512 + (H,W original)
        # Ici on suppose img_rgb_full est bien l'image originale RGB (H_full, W_full, 3).

        print(f"Original size: H={H_full}, W={W_full} | tensor512: {img_tensor_512.shape}")

        # --- B) Labelling (leaf names) sur 512
        print("Extracting ordered texts...")
        leaf_names = extract_ordered_texts_from_image(img_tensor_512.repeat(3, 1, 1), model=self.labelling_model)
        print("Leaf names:", leaf_names)

        # --- C) Cropping sur 512, récupère bbox en coords 512
        print("Cropping tree (predict bbox on 512)...")
        cropped_512_np, bbox_512 = extract_tree_crop_from_image(
            img_tensor_512.repeat(3, 1, 1),
            model=self.cropping_model,
            device=device,
            return_bbox=True
        )
        x0_512, y0_512, x1_512, y1_512 = bbox_512.tolist()

        # --- D) Remap bbox 512 -> bbox originale
        sx = W_full / 512.0
        sy = H_full / 512.0

        x0 = int(max(0, min(W_full - 1, round(x0_512 * sx))))
        y0 = int(max(0, min(H_full - 1, round(y0_512 * sy))))
        x1 = int(max(0, min(W_full,     round(x1_512 * sx))))
        y1 = int(max(0, min(H_full,     round(y1_512 * sy))))

        # sécurité bbox
        if x1 <= x0 + 1 or y1 <= y0 + 1:
            # fallback: pas de crop
            print("WARNING: invalid bbox after remap, fallback to full image.")
            x0, y0, x1, y1 = 0, 0, W_full, H_full

        bbox_full = [x0, y0, x1, y1]
        print("BBox full:", bbox_full)

        # --- E) Crop l'image originale (important pour coords)
        img_crop = img_rgb_full[y0:y1, x0:x1].copy()
        # --- FIX: forcer crop à dimensions paires (évite 408 vs 409)
        # --- FIX robuste: crop aux multiples de 32
        h, w = (y1 - y0), (x1 - x0)
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        x1 = x0 + max(32, new_w)
        y1 = y0 + max(32, new_h)
        img_crop = img_rgb_full[y0:y1, x0:x1].copy()


        print("Cropped original:", img_crop.shape)

        # --- F) Denoising sur le crop original
        print("Denoising crop...")
        cleaned_tensor = denoise_image_tensor(
            tensor_to_gray(img_to_tensor(img_crop)),
            model=self.denoising_model
        )
        cleaned_tensor = cleaned_tensor.repeat(3, 1, 1)
        cleaned_rgb = tensor_to_np(cleaned_tensor)
        print("Cleaned crop:", cleaned_rgb.shape)

       

        # --- G) Node detection sur le crop
        print("Detecting nodes on crop...")
        nodes, corners, leaves = infer_image(cleaned_rgb, conf_threshold, self.nodesdetection_model)
        print(f"Detected (crop): {len(nodes)} nodes, {len(corners)} corners, {len(leaves)} leaves")

        def to_xys(items, default_score=1.0):
                    out = []
                    for t in items:
                        # t peut être [x,y], [x,y,score], [x,y,score,...]
                        x = float(t[0])
                        y = float(t[1])
                        s = float(t[2]) if isinstance(t, (list, tuple)) and len(t) >= 3 else float(default_score)
                        out.append((x, y, s))
                    return out

        nodes   = to_xys(nodes, default_score=1.0)
        corners = to_xys(corners, default_score=1.0)
        leaves  = to_xys(leaves, default_score=1.0)
        
        # --- H) Postprocess correction (toujours dans repère crop)
        print(nodes)
        print("Correcting nodes...")
        nodes, corners, leaves = correction(nodes, corners, leaves, cleaned_rgb.shape[1])
        print(f"After correction (crop): {len(nodes)} nodes, {len(corners)} corners, {len(leaves)} leaves")

        nodes_xy   = [[float(t[0] + x0), float(t[1] + y0)] for t in nodes]
        corners_xy = [[float(t[0] + x0), float(t[1] + y0)] for t in corners]
        leaves_xy  = [[float(t[0] + x0), float(t[1] + y0)] for t in leaves]


        # --- J) Return without newick
        return {
            "image_path": image_path,
            "original_size": {"H": H_full, "W": W_full},
            "bbox_full": bbox_full,         # [x0,y0,x1,y1] dans repère original
            "leaf_names": leaf_names,       # ["A","B",...]
            "nodes": nodes_xy,              # [[x,y], ...] repère original
            "corners": corners_xy,
            "leaves": leaves_xy,
        }
    
    def convert(self, image_path, device="cpu"):
        """
        Méthode requise par la classe abstraite Model.
        On garde une signature compatible, mais on ne build PAS le Newick ici.
        """
        return self.convert_points(image_path=image_path, device=device)

