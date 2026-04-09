import os
import torch
import numpy as np
from PIL import Image
import argparse
from depth_anything_3.api import DepthAnything3

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./GOO_Dataset/images/train")
parser.add_argument('--save_path', type=str, default="./GOO_Dataset/data")

args = parser.parse_args()

def main(data_path, save_path):
    #load depth model
    depth_model = DepthAnything3.from_pretrained("depth-anything/da3-small")
    depth_model = depth_model.to("cuda" if torch.cuda.is_available() else "cpu")

    maps = []
    for i, img_name in enumerate(os.listdir(data_path)):
        img_path = os.path.join(data_path, img_name)
        img = Image.open(img_path).convert("RGB")
        results = depth_model.inference([img])
        depth = results.depth
        depth_tensor = torch.tensor(depth)
        if depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(1)
        elif depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        resized_depth_map = torch.nn.functional.interpolate(depth_tensor, (64, 64), mode='bilinear', align_corners=False).squeeze()
        maps.append(resized_depth_map.cpu().numpy())
        print("Processed image {}/{}".format(i+1, len(os.listdir(data_path))))
     
    np.savez(os.path.join(save_path, "depth_maps.npz"), depth_maps=np.stack(maps))
    print("Saved depth maps to {}".format(os.path.join(save_path, "depth_maps.npz")))

if __name__ == "__main__":
    main(args.data_path, args.save_path)
