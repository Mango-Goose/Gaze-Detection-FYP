import os
import json
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../Dataset")
parser.add_argument("--img_path", type=str, default="../Dataset/images")
args = parser.parse_args()

def main(DATA_PATH, IMG_PATH):
    data = json.load(open(os.path.join(DATA_PATH, "data.json"), "r"))

    #all done for test
    TEST_FRAMES = []

    for i, row in enumerate(data):
        img = Image.open(os.path.join(IMG_PATH, f"{i}.png"))
        
        #check for inout and and set accordingly when needed
        #parse gaze2D to get potential -1 values 
        gaze = row['gaze2D'][1:-1].split(", ")
        gazex = float(gaze[0])
        gazey = float(gaze[1])

        if gazex == -1 or gazey == -1:
            inout = 0
        else:
            inout = 1

        xmin, ymin, xmax, ymax = row['headBbox'][1:-1].split(", ")
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        heads = []
        heads.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'bbox_norm': [(xmin / float(row['width'])), (ymin / float(row['height'])), (xmax / float(row['width'])), (ymax / float(row['height']))],
            'inout': inout,
            'gazex': [gazex],
            'gazey': [gazey],
            'gazex_norm': [gazex / float(row['width'])],
            'gazey_norm': [gazey / float(row['height'])],
            'head_id': 1,
        })
        TEST_FRAMES.append({
            'path': row['path'],
            'heads': heads,
            'num_heads': 1,
            'width': row['width'],
            'height': row['height'],
        })

    out_file = open(os.path.join(DATA_PATH, "test_preprocessed.json"), "w")
    json.dump(TEST_FRAMES, out_file)

if __name__ == "__main__":
    main(args.data_path, args.img_path)