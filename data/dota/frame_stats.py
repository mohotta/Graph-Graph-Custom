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
    parser.add_argument('--in_dir', dest='in_dir', help='The directory to the Dataset', type=str, default="data/dota/new_videos")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str, default="data/dota/frames_stats")

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def get_frames_stats(video_path):

    print("processing: ", video_path)

    frame_stats = []

    capture = cv2.VideoCapture(video_path)
    frame_index = 0

    while capture.isOpened():

        ret, frame = capture.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_stats.append(np.array([height, width]))
        frame_index += 1

    capture.release()

    frame_stats = np.array(frame_stats)

    print(frame_stats.shape)

    return frame_stats

if __name__ == "__main__":

    args = parse_args()
    input_dir = args.in_dir
    output_dir = args.out_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training", "negative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training", "positive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "negative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "positive"), exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            frame_stats = get_frames_stats(os.path.join(root, file))
            out_file = os.path.join(output_dir, root.split("/")[-2], root.split("/")[-1], file[:-4] + ".npy")
            print("save to:", out_file)
            np.save(out_file, frame_stats)

    print("done!")