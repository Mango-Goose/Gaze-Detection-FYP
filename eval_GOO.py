<<<<<<< Updated upstream
=======
import argparse
import torch
import numpy as np
import json
import os 
import numpy as np
from tqdm import tqdm

from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2
from gazelle.dataloader import GazeDataset, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./Dataset2.0")
parser.add_argument("--model", type=str, default="gazelle_dinov2_vitb14")
parser.add_argument("--checkpoint", type=str, default="./experiments/train_my/2026-03-28_16-48-47/epoch_14.pt")
parser.add_argument("--batch_size", type=int, default=60)
args = parser.parse_args()

#Evaluation loop adapted from GazeLLE, specifically the GazeFollow evaluation loop.

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, transform = get_gazelle_model(args.model)
    model.load_gazelle_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(device)

    eval_dataset = GazeDataset('GOO', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    avg_l2s = []
    min_l2s = []
    aucs = []

    for cur_iter, batch in enumerate(eval_dl):
        imgs, bboxes, gazex, gazey, inout, heights, widths = batch

        with torch.no_grad():
            preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds["heatmap"]).squeeze(dim=1)
        for i in range(heatmap_preds.shape[0]):
                
            auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
            avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
            aucs.append(auc)
            avg_l2s.append(avg_l2)
            min_l2s.append(min_l2)

        #average metrics
    epoch_avg_l2 = np.mean(avg_l2s)
    epoch_min_l2 = np.mean(min_l2s)
    epoch_auc = np.mean(aucs)

        #print
    print(" AUC={}, Min L2={}, Avg L2={}".format( round(epoch_auc, 4), round(epoch_min_l2, 4), round(epoch_avg_l2, 4)))


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
