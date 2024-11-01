import argparse
import os
import json
import cv2

"""
    process videos to output videos of given frame count (50) and 
    save time to accident according to that
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', dest='video_dir', help='The directory to the Dataset', type=str, default="data/dota/videos")
    parser.add_argument('--label_dir', dest='label_dir', help='The directory to the Dataset', type=str, default="data/dota/annotations")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str, default="data/dota/processed_videos")
    parser.add_argument('--out_label_dir', dest='out_label_dir', help='The directory to the output files.', type=str, default="data/dota/toas")

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    args = parse_args()
    video_dir = args.video_dir
    annotation_dir = args.label_dir
    output_dir = args.out_dir
    toa_dir = args.out_label_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)
    os.makedirs(toa_dir, exist_ok=True)

    for root, _, files in os.walk(video_dir):
        for file in files:

            print("processing:", file)

            phase = root.split('/')[-1]

            file_path = os.path.join(root, file)
            json_file = os.path.join(annotation_dir, file[:-4] + ".json")
            output_file = os.path.join(output_dir, phase, file)

            if os.path.exists(json_file):

                data = read_json(json_file)
                length = int(data["num_frames"])
                accident_frame = int(data["anomaly_start"])

                print("length:", length)
                print("toa:", accident_frame)

                if length >= 50:

                    cap = cv2.VideoCapture(file_path)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = 10
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

                    if length == 50:
                        toa = accident_frame
                        start_frame = 0
                    elif accident_frame < 40:
                        toa = accident_frame
                        start_frame = 0
                    else:
                        if (accident_frame - 40 + 50) < length:
                            start_frame = accident_frame - 40
                            toa = 40
                        else:
                            start_frame = (length - 50 - 1)
                            toa = accident_frame + start_frame + 1

                    print("start:", start_frame)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    for frame_num in range(start_frame, start_frame+50):
                        ret, frame = cap.read()
                        if ret:
                            video.write(frame)
                        else:
                            break

                    with open(os.path.join(toa_dir, file[:-4] + ".txt"), "w") as f:
                        f.write(str(toa))

                    print("saved:", output_file)

                cap.release()
                video.release()
                cv2.destroyAllWindows()
