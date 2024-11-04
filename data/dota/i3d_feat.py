import argparse
import os
import numpy as np
import cv2

"""
    generate frame stats for videos
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', dest='video_dir', help='The directory to the Dataset', type=str, default="data/dota/new_videos")
    parser.add_argument('--feature_dir', dest='feature_dir', help='The directory to the Dataset', type=str, default="data/dota/obj_feat")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str, default="data/dota/i3d_feat")

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    video_dir = args.video_dir
    feat_dir = args.feature_dir
    output_dir = args.out_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training", "negative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training", "positive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "negative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "positive"), exist_ok=True)

    for root, _, files in os.walk(video_dir):
        for file in files:
            print("processing", file)
            features = np.load(os.path.join(feat_dir, root.split("/")[-2], file[:-4] + ".npz"))["data"]
            out_file = os.path.join(output_dir, root.split("/")[-2], root.split("/")[-1], file[:-4] + ".npy")
            np.save(out_file, features[:, 0, :].reshape(50, 4096))
            print("saved", out_file)

    print("done!")