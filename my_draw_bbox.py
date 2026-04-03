import cv2
import numpy as np
import re
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./Dataset2.0/data.json")
parser.add_argument("--img_path", type=str, default="./Dataset2.0/images/")
parser.add_argument("--image_num", type=str, default="1100")
args = parser.parse_args()


def main(DATA_PATH, IMG_PATH, IMG_NUM):

    #loading up image and dataset
    dataset = json.load(open(DATA_PATH, "r"))
    image = cv2.imread(os.path.join(IMG_PATH, f"{int(IMG_NUM)}.png"))
    bboxes =  dataset[int(IMG_NUM)]['bbox2D']

    #drawing all bboxes - green for products, red for head
    for i in range(0, len(bboxes)):
        head_bboxes = bboxes[i][1:-1].split(", ")
        head_bboxes = [float(coord) for coord in head_bboxes]
        xmin = int(head_bboxes[0])
        ymin = int(head_bboxes[3])
        xmax = int(head_bboxes[2])
        ymax = int(head_bboxes[1])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    head_bboxes = dataset[int(IMG_NUM)]['headBbox'][1:-1].split(", ")
    print(head_bboxes)
    head = [float(coord) for coord in head_bboxes]
    cv2.rectangle(image, (int(head[0]),int(head[1])), (int(head[2]), int(head[3])), (0, 0, 255), 1)

    # Add this before the loop to test if drawing works at all
    #cv2.rectangle(image, (460, 193), (558, 287), (255, 0, 0), 3)  # thick blue box at exact first bbox coords
    cv2.imshow("Image with Head Bounding Box", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(args.data_path, args.img_path, args.image_num)