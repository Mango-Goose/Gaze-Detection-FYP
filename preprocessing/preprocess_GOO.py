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

    for i in range(1):
        bboxes = re.findall(r"\[(.*?)\]", dataset['bboxes'][i][1:-1])
        labels = dataset['labels'][i][1:-1].split(", ") #getting both bboxes and labels as lists

        l_idx = labels.index('25') 
        head_bbox = bboxes[l_idx]  
        
        bboxes = bboxes.remove(bboxes[l_idx]) 
        labels = labels.remove('25')
        


if __name__ == "__main__":
    main(args.data_path)