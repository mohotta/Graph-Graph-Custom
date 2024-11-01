import os
import numpy as np


class DADDataset:
    def __init__(self, data_path, feature, phase='training', vis=False):
        self.data_path = os.path.join(data_path, feature + '_features')
        self.feature = feature
        self.phase = phase
        self.toTensor = False
        # self.device = device
        self.vis = vis
        self.n_frames = 100
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)

        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s" % (filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def read_dad_npz(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['data']  # 100 x 20 x 4096
            labels = data['labels']  # 2
            detections = data['det']  # 100 x 19 x 6
        except:
            raise IOError('Load data error! File: %s' % (data_file))
        # if labels[1] > 0:
        #     toa = [90.0]
        # else:
        #     toa = [self.n_frames + 1]

        print(data_file, "\n",labels)

        # graph_edges, edge_weights = generate_st_graph(detections)

        # if self.toTensor:
        #     features = torch.Tensor(features).to(self.device)  # 100 x 20 x 4096
        #     labels = torch.Tensor(labels).to(self.device)
        #     graph_edges = torch.Tensor(graph_edges).long().to(self.device)
        #     edge_weights = torch.Tensor(edge_weights).to(self.device)
        #     toa = torch.Tensor(toa).to(self.device)

        # if self.vis:
        #     video_id = str(data['ID'])[5:11]  # e.g.: b001_000490_*
        #     return features, labels, graph_edges, edge_weights, toa, detections, video_id
        # else:
        #     return features, labels, graph_edges, edge_weights, toa

if __name__ == "__main__":
    dataset = DADDataset("data/dad", "vgg16", "training")
    for i in range(20):
        dataset.read_dad_npz(i)

