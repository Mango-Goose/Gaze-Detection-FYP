import os
import json
import re
import io
import pandas as pd
from PIL import Image
import argparse
import datasets
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../GOO_Dataset/data")
parser.add_argument("--img_path", type=str, default="../GOO_Dataset/images")
args = parser.parse_args()

#Pitch, Yaw and Roll are rotation angles (roll ear to shoulder, yaw shake head no, pitch nod yes) 

#JSON only really needs to contain the head bbox and gaze target coordinates (including inout), as well as the path to the image.


def main(DATA_PATH, IMG_PATH):

    #TEST
    
    dataset= load_dataset("markytools/goosyntheticv3", split="test")
    print("loading dataset from local path")

    #need to differentiate between person bounding boxes and product bounding boxes - label of 25
    TEST_FRAMES = []

    for i, row in enumerate(dataset):

        #get file location
        img = row['image']
        path = os.path.join(IMG_PATH, "test")
        img.save(os.path.join(path, f"{i}.png"))

        # Store path relative to GOO_Dataset/data/ for proper joining in dataloader
        location = os.path.relpath(os.path.join(path, f"{i}.png"), DATA_PATH)
        width, height = img.size

        #get head bbox
        bboxes = re.findall(r"\[(.*?)\]", row['bboxes'][1:-1])
        labels = row['labels'][1:-1].split(", ") #getting both bboxes and labels as lists

        l_idx = labels.index('25') 
        head_bbox = bboxes[l_idx]  
        
        if row['gazeIdx'] == -1 or row['gaze_cx'] == -1 or row['gaze_cy'] == -1:
            inout = 0
            if row['gaze_cx'] == -1:
                gazex = 0
                gazey = 0
        else:
            inout = 1
            gazex = row['gaze_cx']
            gazey = row['gaze_cy']

        #xmin, xmax, ymin, ymax etc
        xmin, ymin, xmax, ymax = head_bbox[1:-1].split(", ")
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        #checking if gaze is in or out of frame

        heads = []
        heads.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'bbox_norm' : [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
            'inout' : inout, #find out which one means in-frame and set it for all - there is not a label for this in GOO
            'gazex' : [gazex],
            'gazey' : [gazey],
            'gazex_norm' : [gazex / float(width)],
            'gazey_norm' : [gazey / float(height)],
            'head_id' : 1, #need to change this to 1 - only one head per image in GOO.
        
        })
        TEST_FRAMES.append({
            'path' : location,
            'heads' : heads,
            'num_heads' : 1, #always one head per image in GOO
            'width' : width,
            'height' : height,
        })

    #create file to write edited dataset into
    out_file = open(os.path.join(DATA_PATH, "test_preprocessed.json"), "w")
    json.dump(TEST_FRAMES, out_file)

    #TRAIN
    dataset= load_dataset("markytools/goosyntheticv3", split="train")
    print("loading dataset from local path")

    #need to differentiate between person bounding boxes and product bounding boxes - label of 25
    TRAIN_FRAMES = []

    for i, row in enumerate(dataset):

        #get file location
        img = row['image']
        path = os.path.join(IMG_PATH, "train")
        img.save(os.path.join(path, f"{i}.png"))

        # Store path relative to GOO_Dataset/data/ for proper joining in dataloader
        location = os.path.relpath(os.path.join(path, f"{i}.png"), DATA_PATH)
        width, height = img.size

        #get head bbox
        bboxes = re.findall(r"\[(.*?)\]", row['bboxes'][1:-1])
        labels = row['labels'][1:-1].split(", ") #getting both bboxes and labels as lists

        l_idx = labels.index('25') 
        head_bbox = bboxes[l_idx]  
        
        if row['gazeIdx'] == -1 or row['gaze_cx'] == -1 or row['gaze_cy'] == -1:
            inout = 0
            if row['gaze_cx'] == -1:
                gazex = 0
                gazey = 0
        else:
            inout = 1
            gazex = row['gaze_cx']
            gazey = row['gaze_cy']

        #xmin, xmax, ymin, ymax etc
        xmin, ymin, xmax, ymax = head_bbox[1:-1].split(", ")
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        heads = []
        heads.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'bbox_norm' : [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
            'inout' : inout, #find out which one means in-frame and set it for all - there is not a label for this in GOO
            'gazex' : [gazex],
            'gazey' : [gazey],
            'gazex_norm' : [gazex / float(width)],
            'gazey_norm' : [gazey / float(height)],
            'head_id' : 1, #need to change this to 1 - only one head per image in GOO.
        })
        TRAIN_FRAMES.append({
            'path' : location,
            'heads' : heads,
            'num_heads' : 1, #always one head per image in GOO
            'width' : width,
            'height' : height,
        })

    #create file to write edited dataset into
    out_file = open(os.path.join(DATA_PATH, "train_preprocessed.json"), "w")
    json.dump(TRAIN_FRAMES, out_file)


if __name__ == "__main__":
    main(args.data_path, args.img_path)