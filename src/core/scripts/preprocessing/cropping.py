import numpy as np
import torch
import torch.nn as nn
from torchvision import models

class CroppingModel(nn.Module):
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

        self.regressor[2].bias.data = torch.tensor([0.25, 0.25, 0.75, 0.75])

    def forward(self, x):
        feats = self.backbone(x)
        bbox = self.regressor(feats)
        return bbox

def crop_cv2_image_with_bbox(img, bbox):
    """
    Crop an image using a bounding box.

    img  : numpy array (H, W, C)
    bbox : [x_min, y_min, x_max, y_max] in pixels
    """
    H, W = img.shape[:2]

    x_min, y_min, x_max, y_max = map(int, bbox)

    cropped = img[y_min:y_max, x_min:x_max].copy()
    return cropped

def denormalize_bbox(bbox_norm, img_w, img_h):
    """
    Convert normalized bbox [0,1] to pixel coordinates.
    """
    x_min = bbox_norm[0].item() * img_w
    y_min = bbox_norm[1].item() * img_h
    x_max = bbox_norm[2].item() * img_w
    y_max = bbox_norm[3].item() * img_h

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def extract_tree_crop_from_image(
    img_tensor,
    model,
    device="cpu",
    return_bbox=False
):
    """
    Extract a tree crop from an image tensor using a bbox model.

    img_tensor : torch tensor [C, H, W] normalized
    model      : bbox prediction model
    return     : cropped image (numpy), optional bbox
    """
    if img_tensor.ndim != 3:
        raise ValueError("Expected img_tensor shape [C, H, W]")

    print("Preparing image tensor")
    C, H, W = img_tensor.shape

    img_tensor = img_tensor.unsqueeze(0).to(device) # [1, C, H, W]

    print(f"Running bbox model (H={H}, W={W})")
    model.eval()
    with torch.no_grad():
        pred_bbox_norm = model(img_tensor)[0].cpu()

    global_bbox = denormalize_bbox(pred_bbox_norm, W, H)

    print(f"Predicted bbox: {global_bbox.tolist()}")

    print("Cropping image")
    img_np = (
        img_tensor[0]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .cpu()
        .numpy()
    )
    img_np = (img_np * 255).astype(np.uint8)

    cropped_tree = crop_cv2_image_with_bbox(img_np, global_bbox)
    print("Cropping done")

    if return_bbox:
        return cropped_tree, global_bbox

    return cropped_tree
