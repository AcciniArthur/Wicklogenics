import os
import torch
import cv2
from itpt.core import Model
from itpt._data.models.v0.preprocess.denoising import denoise_image_tensor, load_and_preprocess_image, DenoisingModel, img_to_tensor, tensor_to_gray
from itpt._data.models.v0.preprocess.cropping import extract_tree_crop_from_image, CroppingModel
from itpt._data.models.v0.preprocess.labelling import extract_ordered_texts_from_image, LabellingModel, tensor_to_np
from itpt._data.models.v0.nodesdetection.infer_ts_cpu import load_ts_model, infer_image
from itpt._data.models.v0.postprocess.corrector import correction
from itpt._data.models.v0.postprocess.newick import build_newick

class v0(Model):
    def __init__(self):
        super().__init__()
        self._metadata["name"] = "V0"
        self._metadata["description"] = "Version 0"
        self._metadata["version"] = 0
        self.labelling_model = LabellingModel()
        self.cropping_model = CroppingModel()
        self.denoising_model = DenoisingModel()
        self.nodesdetection_model = DenoisingModel()

    def load(self, device="cpu"):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.labelling_model.load_state_dict(torch.load(os.path.join(current_dir, "weights/labelling_model.pth"), map_location=device))
        self.cropping_model.load_state_dict(torch.load(os.path.join(current_dir, "weights/cropping_model.pth"), map_location=device))
        self.denoising_model.load_state_dict(torch.load(os.path.join(current_dir, "weights/denoising_model.pth"), map_location=device))

        self.nodesdetection_model = load_ts_model("./itpt/_data/models/v0/weights/nodesdetection_model.pt")

        self.labelling_model.eval()
        self.cropping_model.eval()
        self.denoising_model.eval()

        print(f"Models loaded")
        self._loaded = True

    def convert(self, image_path, device="cpu"):
        self.ensure_loaded()

        print("Loading and preprocessing image...")
        img_rgb, img_tensor, (H, W) = load_and_preprocess_image(image_path, size=(512, 512))
        print(f"Image loaded: original size (H={H}, W={W}), tensor shape: {img_tensor.shape}")

        print("Extracting ordered texts...")
        ordered_texts = extract_ordered_texts_from_image(img_tensor.repeat(3, 1, 1), model=self.labelling_model)
        print("Ordered texts extracted:", ordered_texts)

        #print("Extracting tree crop...")
        #cropped_tree_np = extract_tree_crop_from_image(img_tensor.repeat(3, 1, 1), model=self.cropping_model)
        #print("Tree crop obtained, shape:", cropped_tree_np.shape)

        #cropped_tree_np_resized = cv2.resize(cropped_tree_np, (512, 512), interpolation=cv2.INTER_LINEAR)
        #print("Resized tree crop:", cropped_tree_np_resized.shape)

        print("Cleaning tree image tensor...")
        cleaned_tensor = denoise_image_tensor(tensor_to_gray(img_to_tensor(img_rgb)), model=self.denoising_model)
        print("Cleaned tensor obtained, shape:", cleaned_tensor.shape)

        cleaned_tensor = cleaned_tensor.repeat(3, 1, 1)
        cleaned_rgb = tensor_to_np(cleaned_tensor)

        print("Detecting nodes...")
        nodes, corners, leaves = infer_image(cleaned_rgb, 0.5, self.nodesdetection_model)
        print(f"Detected: {len(nodes)} nodes, {len(corners)} corners, {len(leaves)} leaves")

        print("Correcting nodes...")
        nodes, corners, leaves = correction(nodes, corners, leaves, 1500)
        print(f"Now: {len(nodes)} nodes, {len(corners)} corners, {len(leaves)} leaves")

        nodes   = [t[:2] for t in nodes]
        corners = [t[:2] for t in corners]
        leaves  = [t[:2] for t in leaves]

        print("Building Newick...")
        #newick = build_newick(leaves, nodes, corners, leaf_names=ordered_texts)
        newick = build_newick(leaves, nodes, corners)
        print("Newick built: ", newick)

        print(f"Convertion finished")
        return str(newick)
