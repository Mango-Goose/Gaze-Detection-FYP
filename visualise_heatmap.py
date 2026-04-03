import cv2
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from gazelle.utils import get_heatmap

parser = argparse.ArgumentParser()
parser.add_argument("--image_num", type=int, default=19)
parser.add_argument("--image_path", type=str, default="./Dataset2.0/images")
parser.add_argument("--data_path", type=str, default="./Dataset2.0/test_preprocessed.json")
args = parser.parse_args()

def main(IMG_NUM, IMG_PATH, DATA_PATH):
    dataset = json.load(open(DATA_PATH, "r"))
    image = cv2.imread(os.path.join(IMG_PATH, f"{int(IMG_NUM)}.png"))

    img_height, img_width = image.shape[:2]
    heatmap = get_heatmap(dataset[int(IMG_NUM)]['heads'][0]['gazex_norm'][0], dataset[int(IMG_NUM)]['heads'][0]['gazey_norm'][0], 64, 64)
    
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
    main(args.image_num, args.image_path, args.data_path)