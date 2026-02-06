import cv2
import numpy as np
import torch
import torchvision
import torchvision.extension


def mask_centroid(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def outputs_to_points(boxes_xyxy, scores, classes, masks_u8):
    """
    boxes_xyxy: [N,4] float
    scores: [N] float
    classes: [N] int
    masks_u8: [N,H,W] uint8/bool or None
    Returns: List[(cls, cx, cy, score)] in resized image coords (if resized)
    """
    n = int(scores.shape[0]) if scores is not None else 0
    if n == 0:
        return []

    pts = []
    has_masks = masks_u8 is not None and len(masks_u8) == n

    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i]
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        if has_masks:
            m = masks_u8[i]
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            if m.max() > 1:
                m = (m > 0).astype(np.uint8)
            c = mask_centroid(m)
            if c is not None:
                cx, cy = c

        pts.append((int(classes[i]), cx, cy, float(scores[i])))

    return pts


def load_ts_model(ts_path):
    torchvision.extension._assert_has_ops()
    model = torch.jit.load(ts_path, map_location="cpu").eval()
    return model


def infer_image(img, score_thresh: float, ts_model):
    # Resize fixe 1500x1500
    img_rs = cv2.resize(img, (1500, 1500), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(img_rs).to("cpu")

    with torch.no_grad():
        out = ts_model(x)

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    classes = out["classes"].detach().cpu().numpy()
    masks = out["masks"].detach().cpu().numpy() if "masks" in out else None

    keep = scores >= float(score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    if masks is not None:
        masks = masks[keep]

    pts = outputs_to_points(boxes, scores, classes, masks)

    leaves, internal_nodes, corners = [], [], []
    for cls, xpt, ypt, sc in pts:
        if cls == 0:
            leaves.append((int(round(xpt)), int(round(ypt)), float(sc)))
        elif cls == 1:
            internal_nodes.append((int(round(xpt)), int(round(ypt)), float(sc)))
        elif cls == 2:
            corners.append((int(round(xpt)), int(round(ypt)), float(sc)))

    return internal_nodes, corners, leaves
