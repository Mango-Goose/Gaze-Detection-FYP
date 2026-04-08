import argparse
import numpy as np
import os
import sys
import random
from datetime import datetime
import torch
import torch.nn as nn
#import wandb - might use weights and biases later, but not curently

# Add parent directory to path to allow imports from gazelle

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()


parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14")
parser.add_argument('--data_path', type=str, default="./screenshots")
parser.add_argument('--dataset', type=str, default='my')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_my')
parser.add_argument('--log_iter', type=int, default=1, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=3)
args = parser.parse_args()

#code adapted from GazeLLE, specifically the GazeFollow training loop.

def main():
    print(f"Starting training for {args.dataset} dataset with args:")

    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)

    model, transform = get_gazelle_model(args.model)

    #check to see if cuda available, if not use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    #utilises PyTorch Dataloaders
    train_dataset = GazeDataset('my', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    #eval_dataset = GazeDataset('my', args.data_path, 'test', transform)
    #eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    #loss - start with using BCE as its what GazeLLE uses but can also try change it up and see if other functions are better
    loss_fn = nn.BCELoss()

    #gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0
    best_epoch = None

    for epoch in range(args.max_epochs):
        #TRAIN EPOCH

        model.train()

        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()

            preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)

            loss = loss_fn(heatmap_preds, heatmaps.to(device))
            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))
            
        scheduler.step()

        ckpt_path = os.path.join(exp_dir, "epoch_{}.pt".format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        #Eval epoch
        print ("running eval epoch")

        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []

    print("Completed training")

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()