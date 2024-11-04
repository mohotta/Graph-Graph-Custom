import os

base = "data/dota/obj_feat"

for _, dirs, _ in os.walk(base):
    if len(dirs) > 0:
        for dir in dirs:
            print("processing", dir)
            with open(f"splits_dota/{'train' if dir == 'training' else 'test'}_split.txt", "w") as f:
                for _, _, files in os.walk(os.path.join(base, dir)):
                    f.write("\n".join(files))
print("done")