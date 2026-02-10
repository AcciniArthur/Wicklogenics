import os
import sys
import cv2
import torch

from src.core.scripts.modelPipeline import Model
from src.core.scripts.preprocessing.denoising import (
    denoise_image_tensor, DenoisingModel,
    img_to_tensor, tensor_to_gray
)
from src.core.scripts.preprocessing.labelling import (
    extract_ordered_texts_from_image, LabellingModel, tensor_to_np
)
from src.core.scripts.nodesdetection.infer_ts_cpu import load_ts_model, infer_image
from src.core.scripts.postprocessing.corrector import correction


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


MODELS_DIR = os.path.abspath(resource_path("../models"))


# ------------------------------------------------------------
# Pipeline v0 (NO CROPPING)
# ------------------------------------------------------------

class v0(Model):
    def __init__(self):
        super().__init__()
        self._metadata.update({
            "name": "V0",
            "description": "Pipeline without cropping",
            "version": 0,
        })

        self.labelling_model = LabellingModel()
        self.denoising_model = DenoisingModel()
        self.nodesdetection_model = None

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    def load(self, device="cpu"):
        self.labelling_model.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, "labelling_model.pth"), map_location=device)
        )
        self.denoising_model.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, "denoising_model.pth"), map_location=device)
        )

        self.nodesdetection_model = load_ts_model(
            os.path.join(MODELS_DIR, "nodesdetection_model.pt")
        )

        self.labelling_model.eval()
        self.denoising_model.eval()

        print("✔ Models loaded (no cropping)")
        self._loaded = True

    # --------------------------------------------------------
    # Main pipeline
    # --------------------------------------------------------

    def convert_points(self, image_path: str, device="cpu", conf_threshold=0.5):
        self.ensure_loaded()

        # ====================================================
        # 1) Load ORIGINAL image (truth reference)
        # ====================================================
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        print("ORIGINAL IMAGE:", (W, H))

        # ====================================================
        # 2) Labelling on resized 512 (names only)
        # ====================================================
        img_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_tensor_512 = torch.from_numpy(img_512).permute(2, 0, 1).float() / 255.0

        leaf_names = extract_ordered_texts_from_image(
            img_tensor_512, model=self.labelling_model
        )
        print("Leaf names:", leaf_names)

        # ====================================================
        # 3) Denoising on ORIGINAL image (no resize)
        # ====================================================
        cleaned_tensor = denoise_image_tensor(
            tensor_to_gray(img_to_tensor(img_rgb)),
            model=self.denoising_model
        )
        cleaned_tensor = cleaned_tensor.repeat(3, 1, 1)
        cleaned_rgb = tensor_to_np(cleaned_tensor)

        det_h, det_w = cleaned_rgb.shape[:2]
        print("DENOISED IMAGE:", (det_w, det_h))

        # ====================================================
        # 4) Node detection (FULL IMAGE)
        # ====================================================
        nodes, corners, leaves = infer_image(
            cleaned_rgb, conf_threshold, self.nodesdetection_model
        )

        print(
            f"Detected: {len(nodes)} nodes, "
            f"{len(corners)} corners, "
            f"{len(leaves)} leaves"
        )

        # ====================================================
        # 5) Normalize to (x,y,score) in IMAGE SPACE
        # ====================================================
        def normalize(items, default_score=1.0):
            out = []
            for t in items:
                x = float(t[0])
                y = float(t[1])
                s = float(t[2]) if len(t) >= 3 else default_score
                out.append((x, y, s))
            return out

        nodes   = normalize(nodes)
        corners = normalize(corners)
        leaves  = normalize(leaves)

        # ====================================================
        # 6) Correction (FULL IMAGE SPACE)
        # ====================================================
        nodes, corners, leaves = correction(
            nodes, corners, leaves, img_width=W
        )

        # ====================================================
        # 7) Strip score, keep pixel coords
        # ====================================================
        nodes_xy   = [[x, y] for x, y, _ in nodes]
        corners_xy = [[x, y] for x, y, _ in corners]
        leaves_xy  = [[x, y] for x, y, _ in leaves]

        return {
            "image_path": image_path,
            "original_size": {"H": H, "W": W},
            "leaf_names": leaf_names,
            "nodes": nodes_xy,
            "corners": corners_xy,
            "leaves": leaves_xy,
        }

    # --------------------------------------------------------
    # Required by abstract Model
    # --------------------------------------------------------

    def convert(self, image_path, device="cpu"):
        return self.convert_points(image_path=image_path, device=device)
