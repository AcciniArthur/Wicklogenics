import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from paddleocr import PaddleOCR

class LabellingModel(nn.Module):
    def __init__(self):
        super().__init__()

        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

        self.regressor[2].bias.data = torch.tensor([0.7, 0.0, 0.95, 1.0])

    def forward(self, x):
        feats = self.backbone(x)
        bbox = self.regressor(feats)
        return bbox

def tensor_to_np(img_tensor):
    """
    img_tensor : torch tensor [C, H, W] normalized
    return     : numpy uint8 image [H, W, C]
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy() # C,H,W -> H,W,C
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return img_np

def box_coverage(inner_boxes, outer_box):
    """
    inner_boxes : Tensor [N, 4]
    outer_box   : Tensor [1, 4]
    return      : Tensor [N]
    """
    ix_min = torch.maximum(inner_boxes[:, 0], outer_box[0, 0])
    iy_min = torch.maximum(inner_boxes[:, 1], outer_box[0, 1])
    ix_max = torch.minimum(inner_boxes[:, 2], outer_box[0, 2])
    iy_max = torch.minimum(inner_boxes[:, 3], outer_box[0, 3])

    inter_w = (ix_max - ix_min).clamp(min=0)
    inter_h = (iy_max - iy_min).clamp(min=0)
    inter_area = inter_w * inter_h

    inner_area = (
        (inner_boxes[:, 2] - inner_boxes[:, 0]) *
        (inner_boxes[:, 3] - inner_boxes[:, 1])
    ).clamp(min=1e-6)

    coverage = inter_area / inner_area
    return coverage

def extract_ordered_texts_from_image(
    img_tensor,
    model,
    device="cpu",
    coverage_threshold=0.9
):
    """
    img_tensor : torch tensor [C, H, W] normalized
    model      : bbox prediction model
    return     : list of ordered texts
    """
    if img_tensor.ndim != 3:
        raise ValueError("Expected img_tensor shape [C, H, W]")

    print("Preparing image tensor")
    C, H, W = img_tensor.shape

    img_tensor = img_tensor.to(device)

    print(f"Running bbox model (H={H}, W={W})")
    model.eval()
    with torch.no_grad():
        pred_bbox_norm = model(img_tensor.unsqueeze(0))[0].cpu()

    global_bbox = pred_bbox_norm.clone()
    global_bbox[0] *= W
    global_bbox[2] *= W
    global_bbox[1] *= H
    global_bbox[3] *= H
    global_bbox = global_bbox.unsqueeze(0) # [1, 4]

    print(f"Predicted bbox: {global_bbox.squeeze(0).tolist()}")

    print("Running PaddleOCR")
    img_np = tensor_to_np(img_tensor)

    ocr = PaddleOCR(
        lang="en",
        use_doc_unwarping=False,
        ocr_version="PP-OCRv5"
    )

    results = ocr.predict(img_np)

    ocr_boxes = []
    ocr_texts = []

    for line in results:
        rec_boxes = line.get("rec_boxes", [])
        rec_texts = line.get("rec_texts", [])

        for box, txt in zip(rec_boxes, rec_texts):
            ocr_boxes.append(box)
            ocr_texts.append(txt)

    print(f"{len(ocr_boxes)} OCR boxes detected")

    if len(ocr_boxes) == 0:
        return []

    ocr_boxes = torch.from_numpy(
        np.array(ocr_boxes, dtype=np.float32)
    ) # [N, 4]

    coverages = box_coverage(ocr_boxes, global_bbox)

    print("OCR coverage values:")
    print(coverages)

    keep_mask = coverages >= coverage_threshold
    kept_count = int(keep_mask.sum().item())

    print(
        f"Coverage filtering >= {coverage_threshold}: "
        f"{kept_count}/{len(ocr_boxes)} kept"
    )

    kept_boxes = ocr_boxes[keep_mask]
    kept_texts = [t for t, k in zip(ocr_texts, keep_mask) if k]

    if len(kept_boxes) == 0:
        print("No text inside global bbox")
        return []

    print("Sorting texts top to bottom")
    y_mins = kept_boxes[:, 1]
    order = torch.argsort(y_mins)

    ordered_texts = [kept_texts[i] for i in order.tolist()]

    print(f"Extracted and ordered texts: {len(ordered_texts)}")

    return ordered_texts
