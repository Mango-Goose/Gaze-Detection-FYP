import argparse
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./GOO_Dataset/images/test/")
parser.add_argument('--image_num', type=str, default="6")
parser.add_argument('--output_path', type=str, default="./depth_maps/depth_output_MiDAS_1.png")
args = parser.parse_args()

def main(DATA_PATH, IMG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load MIDAS
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    img_path = os.path.join(DATA_PATH, f"{IMG}.png")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get depth map
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)

    # Resize and return as numpy array
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    print("depth map shape:", prediction.shape)

    # Normalise
    depth_min = prediction.min()
    depth_max = prediction.max()
    depth_normalized = (prediction - depth_min) / (depth_max - depth_min) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    # Save result
    cv2.imwrite(args.output_path, depth_normalized)
    print(f"Depth map saved to: {args.output_path}")

if __name__ == "__main__":
    main(args.data_path, args.image_num)