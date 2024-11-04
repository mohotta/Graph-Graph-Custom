import argparse
import os

import numpy as np
from ultralytics import YOLO

"""
    get YOLOv8 detections from video files
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', dest='video_dir', help='The directory to the Dataset', type=str, default="data/dota/new_videos")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str, default="data/dota/detections")

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    video_dir = args.video_dir
    output_dir = args.out_dir

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO('yolov8s.pt')

    shape_needed = (50, 19, 6)

    for root, _, files in os.walk(video_dir):
        for file in files:

            file_path = os.path.join(root, file)
            output_file = os.path.join(output_dir, file[:-4] + ".npy")

            print("processing:", file_path)

            results = model(file_path)
            detections = []

            if len(results) > 0:
                arr = map(lambda x: x.boxes.data.cpu().numpy().astype(np.float16), results)
                for item in arr:
                    if len(item) > shape_needed[1]:
                        item = item[:shape_needed[1]]
                    else:
                        item = np.pad(item, ((0, shape_needed[1]-item.shape[0]), (0, 0)), mode='constant', constant_values=0)

                    detections.append(item)
                detections = np.array(detections)
            else:
                detections = np.zeros(shape_needed)

            print(detections.shape)
            assert detections.shape == shape_needed

            np.save(output_file, detections)

            print("saved:", output_file)
