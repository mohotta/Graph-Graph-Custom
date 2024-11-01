import os
from os.path import split

import numpy as np
import pickle

class DoTADataset:
    def __init__(self, data_path, feature, phase='train', vis=False):
        self.data_path = data_path
        self.feature = feature
        self.phase = phase
        self.toTensor = False
        self.vis = vis
        self.n_frames = 50
        self.n_obj = 19
        self.fps = 10.0
        self.dim_feature = self.get_feature_dim(feature)

        self.files_list, self.labels_list = self.read_datalist(data_path, phase)

    def length(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def read_datalist(self, data_path, phase):
        # load training set
        list_file = os.path.join(data_path, self.feature + '_features', '%s.txt' % (phase))
        assert os.path.exists(list_file), "file not exists: %s"%(list_file)
        fid = open(list_file, 'r')
        data_files, data_labels = [], []
        for line in fid.readlines():
            filename, label = line.rstrip().split(' ')
            data_files.append(filename)
            data_labels.append(int(label))
        fid.close()

        return data_files, data_labels

    def get_toa(self, clip_id):
        # handle clip id like "uXXC8uQHCoc_000011_0" which should be "uXXC8uQHCoc_000011"
        clip_id = clip_id if len(clip_id.split('_')[-1]) > 1 else clip_id[:-2]
        label_file = os.path.join(self.data_path, 'frame_labels', clip_id + '.txt')
        assert os.path.exists(label_file)
        f = open(label_file, 'r')
        label_all = []
        for line in f.readlines():
            label = int(line.rstrip().split(' ')[1])
            label_all.append(label)
        f.close()
        label_all = np.array(label_all, dtype=np.int32)
        toa = np.where(label_all == 1)[0][0]
        toa = max(1, toa)  # time-of-accident should not be equal to zero
        return toa

    def get_dad_npz(self, index):
        data_file = os.path.join(self.data_path, self.feature + '_features', self.files_list[index])

        print(f"processing: {data_file}")

        assert os.path.exists(data_file), "file not exists: %s"%(data_file)
        data = np.load(data_file)
        features = data['features']
        label = self.labels_list[index]
        label_onehot = np.array([0, 1]) if label > 0 else np.array([1, 0])
        # get time of accident
        file_id = self.files_list[index].split('/')[1].split('.npz')[0]
        if label > 0:
            toa = [self.get_toa(file_id)]
        else:
            toa = [self.n_frames + 1]

        # construct graph
        attr = 'positive' if label > 0 else 'negative'
        dets_file = os.path.join(self.data_path, 'detections', attr, file_id + '.pkl')
        assert os.path.exists(dets_file), "file not exists: %s"%(dets_file)
        with open(dets_file, 'rb') as f:
            detections = pickle.load(f)
            detections = np.array(detections)  # 100 x 19 x 6
            # graph_edges, edge_weights = generate_st_graph(detections)
        f.close()

        # if self.toTensor:
        #     features = torch.Tensor(features).to(self.device)          #  100 x 20 x 4096
        #     label_onehot = torch.Tensor(label_onehot).to(self.device)  #  2
        #     graph_edges = torch.Tensor(graph_edges).long().to(self.device)
        #     edge_weights = torch.Tensor(edge_weights).to(self.device)
        #     toa = torch.Tensor(toa).to(self.device)

        # if self.vis:
        #     # file_id = file_id if len(file_id.split('_')[-1]) > 1 else file_id[:-2]
        #     # video_path = os.path.join(self.data_path, 'video_frames', file_id, 'images')
        #     # assert os.path.exists(video_path), video_path
        #     return features, label_onehot, graph_edges, edge_weights, toa, detections, file_id
        # else:
        #     return features, label_onehot, graph_edges, edge_weights, toa

        feat_file = os.path.join(self.data_path, 'obj_feat', "training" if self.phase == "train" else "testing", self.files_list[index].split("/")[-1])

        assert features.shape == (50,20,4096)
        assert label_onehot.shape == (2,)
        assert detections.shape == (50, 19, 6)

        print(label_onehot)

        np.savez_compressed(feat_file, data=features, labels=label_onehot, det=detections, ID=self.files_list[index])

        print(f"saved to: {feat_file}")

if __name__ == "__main__":

    os.makedirs("data/a3d/obj_feat", exist_ok=True)
    os.makedirs("data/a3d/obj_feat/training", exist_ok=True)
    os.makedirs("data/a3d/obj_feat/testing", exist_ok=True)

    for phase in ["train", "test"]:
        dataset = A3DDataset("data/a3d", "vgg16", phase)
        for i in range(dataset.length()):
            dataset.get_dad_npz(i)
