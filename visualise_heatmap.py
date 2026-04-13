import cv2
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from gazelle.utils import get_heatmap
from depth_anything_3.api import DepthAnything3

parser = argparse.ArgumentParser()
parser.add_argument("--image_num", type=int, default=1100)
parser.add_argument("--image_path", type=str, default="./GOO_Dataset/images/test")
parser.add_argument("--data_path", type=str, default="./GOO_Dataset/data/test_preprocessed.json")
parser.add_argument("--depth_path", type=str, default="./GOO_Dataset/data/test_depth_maps.npz")
args = parser.parse_args()

def main(IMG_NUM, IMG_PATH, DATA_PATH, DEPTH_PATH):
    # Initialize depth model
    depth_model = DepthAnything3.from_pretrained("depth-anything/da3-giant")
    depth_model = depth_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = json.load(open(DATA_PATH, "r"))
    image = cv2.imread(os.path.join(IMG_PATH, f"{int(IMG_NUM)}.png"))
    #depth_maps = np.load(os.path.join(DEPTH_PATH))['depth_maps']

    #depth = depth_maps[IMG_NUM]
    results = depth_model.inference([image])
    depth = results.depth
    depth_tensor = torch.tensor(depth)

    if depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.unsqueeze(1)
    elif depth_tensor.ndim == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    resized_depth_map = torch.nn.functional.interpolate(depth_tensor, (64, 64), mode='bilinear', align_corners=False).squeeze()
        

    img_height, img_width = image.shape[:2]
    heatmap = get_heatmap(resized_depth_map, dataset[int(IMG_NUM)]['heads'][0]['gazex_norm'][0], dataset[int(IMG_NUM)]['heads'][0]['gazey_norm'][0], 64, 64)
    
    img = TF.to_pil_image(image)
    h_img = TF.to_pil_image(heatmap)
    
    h_img = h_img.resize((img_width, img_height), Image.BILINEAR)
    h_img = h_img.convert("RGB")
    
    heatmap_array = np.array(h_img)[:, :, 0]
    heatmap_inverted = 255 - heatmap_array
    
    h_img = cv2.applyColorMap(heatmap_inverted, cv2.COLORMAP_JET)
    heatmap_img = Image.fromarray(h_img)

    res = Image.blend(img, heatmap_img, alpha=0.5)
    plt.imshow(res)
    plt.title("Heatmap")
    plt.show()

if __name__ == "__main__":
    main(args.image_num, args.image_path, args.data_path, args.depth_path)