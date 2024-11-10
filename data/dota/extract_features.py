from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import os, cv2
import argparse, sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

CLASSES = ('__background__', 'Car', 'Pedestrian', 'Cyclist')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dota_dir', dest='dota_dir', help='The directory to the Dashcam Accident Dataset', default="data/dota")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', default="data/dota_ego/obj_feat")
    parser.add_argument('--n_frames', dest='n_frames', help='The number of frames sampled from each video', default=50)
    parser.add_argument('--n_boxes', dest='n_boxes', help='The number of bounding boxes for each frame', default=19)
    parser.add_argument('--dim_feat', dest='dim_feat', help='The dimension of extracted ResNet101 features',
                        default=4096)
    #
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def get_video_frames(video_file, n_frames=50):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert counter == n_frames
    return video_data


def bbox_to_imroi(bboxes, image):
    """
    bboxes: (n, 4), ndarray
    image: (H, W, 3), ndarray
    """
    imroi_data = []
    for bbox in bboxes:
        imroi = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
        imroi_data.append(imroi)
    imroi_data = torch.stack(imroi_data)
    return imroi_data


def get_boxes(dets_all, im_size):
    bboxes = []
    for bbox in dets_all:
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        x1 = min(max(0, x1), im_size[1] - 1)  # 0<=x1<=W-1
        y1 = min(max(0, y1), im_size[0] - 1)  # 0<=y1<=H-1
        x2 = min(max(x1, x2), im_size[1] - 1)  # x1<=x2<=W-1
        y2 = min(max(y1, y2), im_size[0] - 1)  # y1<=y2<=H-1
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        if h > 2 and w > 2:  # the area is at least 9
            bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes, dtype=np.int32)
    return bboxes


def extract_features(detections_path, video_path, dest_path, phase):

    for root, _, files in os.walk(video_path):
        for file in files:
            video_file = os.path.join(root, file)
            print("processing:", video_file)
            video_frames = get_video_frames(video_file, n_frames=args.n_frames)
            detections_file = os.path.join(detections_path, file[:-4] + ".npy")
            detections = np.load(detections_file)
            label = np.array([0,1]) if root.split("/")[-1] == "positive" else np.array([1,0])
            feat_file = os.path.join(dest_path, file[:-4] + ".npz")

            features_vgg16 = np.zeros((args.n_frames, args.n_boxes + 1, args.dim_feat), dtype=np.float32)


            for i, frame in tqdm(enumerate(video_frames), total=len(video_frames)):
                bboxes = get_boxes(detections[i], frame.shape)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    image = transform(Image.fromarray(frame))
                    ims_frame = torch.unsqueeze(image, dim=0).float().to(device=device)
                    feature_frame = torch.squeeze(feat_extractor(ims_frame))
                    features_vgg16[i, 0, :] = feature_frame.cpu().numpy() if feature_frame.is_cuda else feature_frame.detach().numpy()

                    # extract object feature
                    if len(bboxes) > 0:
                        # bboxes to roi data
                        ims_roi = bbox_to_imroi(bboxes, frame)  # (n, 3, 224, 224)
                        ims_roi = ims_roi.float().to(device=device)
                        feature_roi = torch.squeeze(torch.squeeze(feat_extractor(ims_roi), dim=-1), dim=-1)  # (4096,)
                        features_vgg16[i, 1:len(bboxes) + 1, :] = feature_roi.cpu().numpy() if feature_roi.is_cuda else feature_roi.detach().numpy()

                    np.savez_compressed(feat_file, data=features_vgg16, det=detections, labels=label, ID=file[:-4])


def run(detections_path, video_path, dest_path):
    # prepare the result paths
    train_path = os.path.join(dest_path, 'training')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(dest_path, 'testing')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # process training set
    extract_features(detections_path, os.path.join(video_path, 'training'), train_path, 'training')
    # process testing set
    extract_features(detections_path, os.path.join(video_path, 'testing'), test_path, 'testing')


if __name__ == "__main__":
    args = parse_args()

    feat_extractor = models.vgg16(pretrained=True)
    feat_extractor.classifier = nn.Sequential(*list(feat_extractor.classifier.children())[:-1])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feat_extractor = feat_extractor.to(device=device)
    feat_extractor.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    detections_path = osp.join(args.dota_dir, 'detections')
    video_path = osp.join(args.dota_dir, 'new_videos')
    run(detections_path, video_path, args.out_dir)

    print("Done!")