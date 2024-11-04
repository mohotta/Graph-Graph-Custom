import numpy as np
import os

base = "data/dota/obj_feat"
video_base = "data/dota/new_videos"

for phase in ["training", "testing"]:
    for root, _, files in os.walk(os.path.join(video_base, phase, "negative")):
        for file in files:
            print("processing", file)
            feat = np.load(os.path.join(base, phase, file[:-4] + ".npz"))
            np.savez_compressed(os.path.join(base, phase, file[:-4] + ".npz"), data=feat["data"], det=feat["det"], labels=np.array([1,0]), ID=file[:-4])

print("done!")