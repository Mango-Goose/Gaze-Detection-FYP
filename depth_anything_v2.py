import argparse
import os
import torch
import numpy as np
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./GOO_Dataset/images/test/")
parser.add_argument('--image_num', type=str, default="6")
parser.add_argument('--output_path', type=str, default="./depth_maps/depth_output_anything_1.png")
args = parser.parse_args()


def main(DATA_PATH, IMG):
    img_path = os.path.join(DATA_PATH, f"{IMG}.png")
    img = Image.open(img_path)
    img = img.convert('RGB')  # Convert to RGB if it's RGBA or other format

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    inputs = image_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    # Normalize depth map to 0-255 and convert to uint8 for saving as PNG
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map_uint8 = depth_map_normalized.astype(np.uint8)

    # Save the depth map
    Image.fromarray(depth_map_uint8, mode='L').save(args.output_path)


if __name__ == "__main__":
    main(args.data_path, args.image_num)