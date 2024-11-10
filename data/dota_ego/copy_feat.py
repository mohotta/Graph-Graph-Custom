import os


def copy(source, dest):
    with open(source, "rb") as f1:
        with open(dest, "wb") as f2:
            f2.write(f1.read())
            f2.close()
        f1.close()

source = "data/dota"
dest = "data/dota_ego"

os.makedirs(os.path.join(dest, "i3d_feat"), exist_ok=True)
os.makedirs(os.path.join(dest, "i3d_feat", "training"), exist_ok=True)
os.makedirs(os.path.join(dest, "i3d_feat", "training", "negative"), exist_ok=True)
os.makedirs(os.path.join(dest, "i3d_feat", "training", "positive"), exist_ok=True)
os.makedirs(os.path.join(dest, "i3d_feat", "testing"), exist_ok=True)
os.makedirs(os.path.join(dest, "i3d_feat", "testing", "negative"), exist_ok=True)
os.makedirs(os.path.join(dest, "i3d_feat", "testing", "positive"), exist_ok=True)

os.makedirs(os.path.join(dest, "obj_feat"), exist_ok=True)
os.makedirs(os.path.join(dest, "obj_feat", "training"), exist_ok=True)
os.makedirs(os.path.join(dest, "obj_feat", "testing"), exist_ok=True)

for root, _, files in os.walk(os.path.join(source, "ego_videos")):
    for file in files:

        print("processing", file)

        i3d_file = os.path.join(source, "i3d_feat", root.split("/")[-2], root.split("/")[-1], file[:-4] + ".npy")
        obj_file = os.path.join(source, "obj_feat", root.split("/")[-2], file[:-4] + ".npz")
        out_i3d = os.path.join(dest, "i3d_feat", root.split("/")[-2], root.split("/")[-1], file[:-4] + ".npy")
        out_obj = os.path.join(dest, "obj_feat", root.split("/")[-2], file[:-4] + ".npz")

        copy(i3d_file, out_i3d)
        copy(obj_file, out_obj)

print("done!")
