import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = nn.functional.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = nn.functional.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))

def tensor_to_gray(tensor, threshold=0.5):
    """
    Convert a tensor image to grayscale.

    tensor : tensor [C, H, W] normalized
    return : tensor [1, H, W] normalized
    """
    if tensor.ndim != 3:
        raise ValueError("Expected tensor shape [C, H, W]")

    gray = tensor.mean(dim=0, keepdim=True)
    #binary = (gray > threshold).float()

    #return binary
    return gray

def img_to_tensor(img):
    """
    img : numpy array [H, W, C]
    return : tensor [C, H, W] normalized
    """
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return img

def load_and_preprocess_image(path, size=(512, 512)):
    """
    Load an image from disk and preprocess it.

    return:
    - img_rgb    : numpy array [H, W, 3]
    - img_tensor : tensor [1, 1, H, W] normalized
    - (H, W)     : original size
    """
    try:
        pil_img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(path)

    img_rgb = np.array(pil_img)
    H, W, _ = img_rgb.shape

    img_resized = cv2.resize(
        img_rgb,
        size,
        interpolation=cv2.INTER_LINEAR
    )

    img_tensor = img_to_tensor(img_resized)
    img_tensor = tensor_to_gray(img_tensor) # [1, H, W]

    return img_rgb, img_tensor, (H, W)

def denoise_image_tensor(img_tensor, model, device="cpu"):
    """
    img_tensor : tensor [C, H, W] normalized
    model      : denoising model
    return     : tensor [C, H, W] normalized
    """
    if img_tensor.ndim != 3:
        raise ValueError("Expected tensor shape [C, H, W]")

    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)

    return pred.squeeze(0).cpu()
