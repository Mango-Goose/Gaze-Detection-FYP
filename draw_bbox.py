import cv2
import numpy as np
import re
import os
import argparse
import datasets
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./GOO_Dataset/data")
parser.add_argument("--img_path", type=str, default="./GOO_Dataset/images/train")
parser.add_argument("--image_num", type=str, default="450")
args = parser.parse_args()


def main(DATA_PATH, IMG_PATH, IMG_NUM):

    #loading up image and dataset
    image = cv2.imread(os.path.join(IMG_PATH, f"{IMG_NUM}.png"))
    dataset= load_dataset(DATA_PATH, split="train")

    bboxes = re.findall(r"\[(.*?)\]", dataset['bboxes'][int(IMG_NUM)][1:-1])
    labels = dataset['labels'][int(IMG_NUM)][1:-1].split(", ")

    #drawing all bboxes - green for products, red for head
    for i in range(len(labels)):
        head_bboxes = bboxes[i][:-2].split(".0, ")

        if labels[i] == '25':
            cv2.rectangle(image, (int(head_bboxes[0]), int(head_bboxes[1])), (int(head_bboxes[2]), int(head_bboxes[3])), (0, 0, 255), 1)
        else:
            cv2.rectangle(image, (int(head_bboxes[0]), int(head_bboxes[1])), (int(head_bboxes[2]), int(head_bboxes[3])), (0, 255, 0), 1)
    
    cv2.imshow("Image with Head Bounding Box", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(args.data_path, args.img_path, args.image_num)