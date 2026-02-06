import os
import json
import re
import pandas as pd
from PIL import Image
import argparse
import datasets
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../GOO_Dataset/data")
args = parser.parse_args()

#Pitch, Yaw and Roll are rotation angles (roll ear to shoulder, yaw shake head no, pitch nod yes) 

#JSON only really needs to contain the head bbox and gaze target coordinates (including inout), as well as the path to the image.


def main(DATA_PATH):
    #TEST

    #check if dataset downloaded, if not stream it from huggingface
    if len(os.listdir(DATA_PATH)) == 0:
        dataset = load_dataset("markytools/goosyntheticv3", split="test", streaming=True)  #issue with this - creates an iterable dataset not a dataset - doesnt act the same.
        print("streaming dataset from huggingface, dataset: ", next(iter(dataset)) )
    else:
        dataset= load_dataset(DATA_PATH, split="test")
        print("loading dataset from local path")

    #need to differentiate between person bounding boxes and product bounding boxes - label of 25
    TEST_FRAMES = []

    for i in range(dataset.num_rows):

        #get file location
        img = dataset['image'][i]
        filename = img.filename
        location = os.path.join(DATA_PATH, filename)
        width, height = img.size

        #get head bbox
        bboxes = re.findall(r"\[(.*?)\]", dataset['bboxes'][i][1:-1])
        labels = dataset['labels'][i][1:-1].split(", ") #getting both bboxes and labels as lists

        l_idx = labels.index('25') 
        head_bbox = bboxes[l_idx]  
        
        bboxes = bboxes.remove(bboxes[l_idx]) 
        labels = labels.remove('25')
        
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
            'inout' : 1, #find out which one means in-frame and set it for all - there is not a label for this in GOO
            'gazex' : dataset['gaze_cx'][i],
            'gazey' : dataset['gaze_cy'][i],
            'gazex_norm' : dataset['gaze_cx'][i],
            'gazey_norm' : dataset['gaze_cy'][i],
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
    #check if dataset downloaded, if not stream it from huggingface
    if len(os.listdir(DATA_PATH)) == 0:
        dataset = load_dataset("markytools/goosyntheticv3", split="train", streaming=True)  #issue with this - creates an iterable dataset not a dataset - doesnt act the same.
        print("streaming dataset from huggingface, dataset: ", next(iter(dataset)) )
    else:
        dataset= load_dataset(DATA_PATH, split="train")
        print("loading dataset from local path")

    #need to differentiate between person bounding boxes and product bounding boxes - label of 25
    TRAIN_FRAMES = []

    for i in range(dataset.num_rows):

        #get file location
        img = dataset['image'][i]
        filename = img.filename
        location = os.path.join(DATA_PATH, filename)
        width, height = img.size

        #get head bbox
        bboxes = re.findall(r"\[(.*?)\]", dataset['bboxes'][i][1:-1])
        labels = dataset['labels'][i][1:-1].split(", ") #getting both bboxes and labels as lists

        l_idx = labels.index('25') 
        head_bbox = bboxes[l_idx]  
        
        bboxes = bboxes.remove(bboxes[l_idx]) 
        labels = labels.remove('25')
        
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
            'inout' : 1, #find out which one means in-frame and set it for all - there is not a label for this in GOO
            'gazex' : dataset['gaze_cx'][i],
            'gazey' : dataset['gaze_cy'][i],
            'gazex_norm' : dataset['gaze_cx'][i],
            'gazey_norm' : dataset['gaze_cy'][i],
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
    main(args.data_path)