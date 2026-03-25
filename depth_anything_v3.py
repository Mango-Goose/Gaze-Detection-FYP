import argparse
import os
import cv2

import torch
import numpy as np
import glob
from PIL import Image
#from transformers import pipeline
from depth_anything_3.api import DepthAnything3

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./GOO_Dataset/images/test/")
parser.add_argument('--image_num', type=str, default="6")
parser.add_argument('--output_path', type=str, default="./depth_maps/depth_output_anything_1.png")
args = parser.parse_args()


def main(DATA_PATH, IMG):
    img_path = os.path.join(DATA_PATH, f"{IMG}.png")
    img = Image.open(img_path)
    img = img.convert('RGB') 

    #depth_estimator  = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
    model = DepthAnything3.from_pretrained("depth-anything/da3-large")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")


    results = model.inference([img])
    depth = results.depth

    # Ensure depth is a numpy array (handle torch tensors or list outputs)
    if hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    if isinstance(depth, (list, tuple)):
        depth = depth[0]

    # If model returns (1, H, W), squeeze to (H, W)
    depth = np.asarray(depth)
    depth = np.squeeze(depth)

    print("depth map shape:", depth.shape)

    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = depth_normalized.astype(np.uint8)

    # Invert the color grading: closer objects become brighter, farther become darker
    depth_image = 255 - depth_image

    # Optionally resize for display (keep depth_image as a NumPy array)
    #depth_display = cv2.resize(depth_image, (1280, 720))

    # Show depth map (grayscale)
    cv2.imshow("Depth Map", depth_image)
    cv2.waitKey(0)

    # Save the depth map as a grayscale PNG
    cv2.imwrite(args.output_path, depth_image)

    


if __name__ == "__main__":
    main(args.data_path, args.image_num)