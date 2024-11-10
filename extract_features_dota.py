import torch
import numpy as np
from models import *
from dataset_dota import *
from torch.utils.data import DataLoader

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.optim.lr_scheduler import MultiStepLR

from torchmetrics.functional import pairwise_cosine_similarity
import scipy.io as io

import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import time
from eval_utils import evaluation

torch.manual_seed(0)  # 3407

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="data/dota/obj_feat", help="Path to extracted objects data")
parser.add_argument("--img_dataset_path", type=str, default="data/dota/i3d_feat", help="Path to I3D feature data")
parser.add_argument("--obj_mapping_file", type=str, default="data/dota/obj_idx_to_labels.json",
                    help="path to object label mapping file")
parser.add_argument("--feature_path", type=str, default="data/dota/gog_features",
                    help="path to gog features")
parser.add_argument("--split_path", type=str, default="splits_dota/", help="Path to train/test split")
# parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
# parser.add_argument("--batch_size", type=int, default=1, help="Size of each training batch for frames")
# parser.add_argument("--video_batch_size", type=int, default=1, help="Size of each training batch for video")
# parser.add_argument("--test_video_batch_size", type=int, default=1, help="Size of each test batch for video")
parser.add_argument("--input_dim", type=int, default=4096, help="input dimension")
parser.add_argument("--img_feat_dim", type=int, default=4096, help="input i3d feature dimension")
parser.add_argument("--embedding_dim", type=int, default=256, help="embedding size of the difference")
parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
# parser.add_argument("--test_only", type=int, default=0, help="Test the loaded model only the video 0(False)/1(True)")
parser.add_argument("--ref_interval", type=int, default=20, help="Interval size for reference frames")
parser.add_argument("--checkpoint_model", type=str, default="model_checkpoints/dota/SpaceTempGoG_detr_dota_3.pth", help="Optional path to checkpoint model")

opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_frames = 50


def main():

    os.makedirs(opt.feature_path, exist_ok=True)
    os.makedirs(os.path.join(opt.feature_path, "training"), exist_ok=True)
    os.makedirs(os.path.join(opt.feature_path, "testing"), exist_ok=True)

    # Define training set
    train_dataset = FeaturesDataset(
        img_dataset_path=opt.img_dataset_path,
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        #		frame_batch_size=opt.batch_size,
        ref_interval=opt.ref_interval,
        objmap_file=opt.obj_mapping_file,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    # Define test set
    test_dataset = FeaturesDataset(
        img_dataset_path=opt.img_dataset_path,
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        #		frame_batch_size=opt.batch_size,
        ref_interval=opt.ref_interval,
        objmap_file=opt.obj_mapping_file,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Define network
    model = SpaceTempGoG_detr_dota(input_dim=opt.input_dim, embedding_dim=opt.embedding_dim,
                                  img_feat_dim=opt.img_feat_dim, num_classes=opt.num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))

    for phase in ["training", "testing"]:
        for batch_i, (
                X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat,
                batch_vec,
                toa, filename) in enumerate(train_dataloader if phase=="training" else test_dataloader):

            print("processing:", filename)

            # Processing the inputs from the dataloader
            X = X.reshape(-1, X.shape[2])
            img_feat = img_feat.reshape(-1, img_feat.shape[2])
            edge_index = edge_index.reshape(-1, edge_index.shape[2])
            edge_embeddings = edge_embeddings.view(-1, edge_embeddings.shape[-1])
            video_adj_list = video_adj_list.reshape(-1, video_adj_list.shape[2])
            temporal_adj_list = temporal_adj_list.reshape(-1, temporal_adj_list.shape[2])
            y = y_true.reshape(-1)
            toa = toa.item()

            # pairwise cosine similarity between the objects
            obj_vis_feat = obj_vis_feat.reshape(-1, obj_vis_feat.shape[-1]).to(device)
            feat_sim = pairwise_cosine_similarity(obj_vis_feat + 1e-7, obj_vis_feat + 1e-7)
            temporal_edge_w = feat_sim[temporal_adj_list[0, :], temporal_adj_list[1, :]]
            batch_vec = batch_vec.view(-1).long()

            X, edge_index, y, img_feat, video_adj_list = X.to(device), edge_index.to(device), y.to(device), img_feat.to(
                device), video_adj_list.to(device)
            temporal_adj_list, temporal_edge_w, edge_embeddings, batch_vec = temporal_adj_list.to(
                device), temporal_edge_w.to(device), edge_embeddings.to(device), batch_vec.to(device)

            # Get predictions from the model
            g_embed, img_feat, frame_embed_sg = model.extract_features(X, edge_index, img_feat, video_adj_list,
                                                                       edge_embeddings, temporal_adj_list,
                                                                       temporal_edge_w, batch_vec)

            np.savez_compressed(os.path.join(opt.feature_path, phase, filename[0] + ".npz"), image_feat=img_feat.cpu().detach().numpy(), graph_level_feat=g_embed.cpu().detach().numpy(), frame_level_feat=frame_embed_sg.cpu().detach().numpy(), id=filename[0])


if __name__ == "__main__":
    main()
    print("done!")
