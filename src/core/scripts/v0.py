import os
import sys
import cv2
import torch
from datetime import datetime

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
# Utils paths (PyInstaller friendly)
# ------------------------------------------------------------

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


MODELS_DIR = os.path.abspath(resource_path("../models"))


def _project_root() -> str:
    """
    Essaie de viser la racine du projet (là où se trouve app.py).
    En dev: cwd est souvent la racine. En exe: ça dépend, mais ça reste OK.
    """
    return os.getcwd()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_rgb(img_rgb, path: str) -> None:
    """img_rgb: numpy (H,W,3) RGB -> écrit en PNG via OpenCV (BGR)."""
    if img_rgb is None:
        return
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def _save_gray(img_gray, path: str) -> None:
    """img_gray: numpy (H,W) grayscale."""
    if img_gray is None:
        return
    cv2.imwrite(path, img_gray)


def _draw_points_on_rgb(img_rgb, nodes, corners, leaves, out_path: str) -> None:
    """
    Dessine un overlay simple pour debug.
    nodes/corners/leaves: list of (x,y,score) or (x,y)
    """
    if img_rgb is None:
        return

    canvas = img_rgb.copy()

    def iter_xy(arr):
        for t in arr:
            if len(t) >= 2:
                yield int(round(t[0])), int(round(t[1]))

    # Colors (RGB)
    col_node = (255, 0, 0)      # red
    col_corner = (255, 255, 0)  # yellow
    col_leaf = (0, 0, 255)      # blue

    for x, y in iter_xy(nodes):
        cv2.circle(canvas, (x, y), 4, col_node[::-1], -1)   # OpenCV expects BGR, but canvas is RGB; we convert at save
    for x, y in iter_xy(corners):
        cv2.circle(canvas, (x, y), 4, col_corner[::-1], -1)
    for x, y in iter_xy(leaves):
        cv2.circle(canvas, (x, y), 4, col_leaf[::-1], -1)

    _save_rgb(canvas, out_path)


# ------------------------------------------------------------
# Pipeline v0 (NO CROPPING) + debug saves
# ------------------------------------------------------------

class v0(Model):
    def __init__(self):
        super().__init__()
        self._metadata.update({
            "name": "V0",
            "description": "Pipeline without cropping + debug images",
            "version": 0,
        })

        self.labelling_model = LabellingModel()
        self.denoising_model = DenoisingModel()
        self.nodesdetection_model = None

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

    def convert_points(self, image_path: str, device="cpu", conf_threshold=0.5):
        self.ensure_loaded()

        # ---- Debug run folder
        base = os.path.join(_project_root(), "saveimg")
        _ensure_dir(base)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, run_id)
        _ensure_dir(run_dir)

        # ====================================================
        # 1) Load ORIGINAL image
        # ====================================================
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        print("ORIGINAL IMAGE:", (W, H))

        #_save_rgb(img_rgb, os.path.join(run_dir, "01_original.png"))

        # ====================================================
        # 2) Labelling on resized 512 (names only)
        # ====================================================
        img_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        #_save_rgb(img_512, os.path.join(run_dir, "02_resize_512_for_labelling.png"))

        img_tensor_512 = torch.from_numpy(img_512).permute(2, 0, 1).float() / 255.0
        leaf_names = extract_ordered_texts_from_image(img_tensor_512, model=self.labelling_model)
        print("Leaf names:", leaf_names)

        # ====================================================
        # 3) Denoising on ORIGINAL image
        # ====================================================
        # Save gray input to denoise
        gray_in_tensor = tensor_to_gray(img_to_tensor(img_rgb))
        gray_in = tensor_to_np(gray_in_tensor.repeat(3, 1, 1))
        #_save_rgb(gray_in, os.path.join(run_dir, "03_gray_input_to_denoise.png"))

        cleaned_tensor = denoise_image_tensor(
            gray_in_tensor,
            model=self.denoising_model
        )
        # cleaned_tensor shape likely [1,H,W] or [H,W]; we normalize to RGB for save
        cleaned_rgb = tensor_to_np(cleaned_tensor.repeat(3, 1, 1))
        det_h, det_w = cleaned_rgb.shape[:2]
        print("DENOISED IMAGE:", (det_w, det_h))

        #_save_rgb(cleaned_rgb, os.path.join(run_dir, "04_denoised.png"))

        # ====================================================
        # 4) Node detection (FULL IMAGE)
        # ====================================================
        nodes, corners, leaves = infer_image(cleaned_rgb, conf_threshold, self.nodesdetection_model)
        print(f"Detected: {len(nodes)} nodes, {len(corners)} corners, {len(leaves)} leaves")

        # overlay on denoised
        #_draw_points_on_rgb(cleaned_rgb, nodes, corners, leaves, os.path.join(run_dir, "05_detected_overlay_on_denoised.png"))
        # overlay on original
        #_draw_points_on_rgb(img_rgb, nodes, corners, leaves, os.path.join(run_dir, "06_detected_overlay_on_original.png"))

        # ====================================================
        # 5) Normalize to (x,y,score) in IMAGE SPACE
        # ====================================================
        def normalize(items, default_score=1.0):
            out = []
            for t in items:
                x = float(t[0])
                y = float(t[1])
                s = float(t[2]) if isinstance(t, (list, tuple)) and len(t) >= 3 else default_score
                out.append((x, y, s))
            return out

        nodes_n = normalize(nodes)
        corners_n = normalize(corners)
        leaves_n = normalize(leaves)

        # save overlays again (normalized)
        #_draw_points_on_rgb(cleaned_rgb, nodes_n, corners_n, leaves_n, os.path.join(run_dir, "07_normalized_overlay_on_denoised.png"))
        #_draw_points_on_rgb(img_rgb, nodes_n, corners_n, leaves_n, os.path.join(run_dir, "08_normalized_overlay_on_original.png"))

        # ====================================================
        # 6) Correction (FULL IMAGE SPACE)
        # ====================================================
        nodes_c, corners_c, leaves_c = correction(nodes_n, corners_n, leaves_n, img_width=W)

        print(f"After correction: {len(nodes_c)} nodes, {len(corners_c)} corners, {len(leaves_c)} leaves")

        # overlays after correction
        #_draw_points_on_rgb(img_rgb, nodes_c, corners_c, leaves_c, os.path.join(run_dir, "09_corrected_overlay_on_original.png"))
        #_draw_points_on_rgb(cleaned_rgb, nodes_c, corners_c, leaves_c, os.path.join(run_dir, "10_corrected_overlay_on_denoised.png"))

       # ====================================================
        # 7) Rescale coordinates: * original_size / 1500
        # ====================================================
        scale_x = W / 1500.0
        scale_y = H / 1500.0

        def rescale(arr):
            out = []
            for x, y, _ in arr:
                out.append([
                    float(x * scale_x),
                    float(y * scale_y),
                ])
            return out

        nodes_xy   = rescale(nodes_c)
        corners_xy = rescale(corners_c)
        leaves_xy  = rescale(leaves_c)



        # final overlay (just in case)
        #_draw_points_on_rgb(img_rgb, nodes_xy, corners_xy, leaves_xy, os.path.join(run_dir, "11_final_overlay_on_original.png"))

        return {
            "image_path": image_path,
            "original_size": {"H": H, "W": W},
            "leaf_names": leaf_names,
            "nodes": nodes_xy,
            "corners": corners_xy,
            "leaves": leaves_xy,
            "debug_run_dir": run_dir,  # utile pour afficher où sont les images
        }

    def convert(self, image_path, device="cpu"):
        return self.convert_points(image_path=image_path, device=device)
