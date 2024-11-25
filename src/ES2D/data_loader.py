import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
import sys
import trimesh
from Get_landmarks import get_landmarks


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        vertices = self.data[index]["vertices"]
        template = self.data[index]["template"]
        land_vertices = self.data[index]["land_vertices"]
        land_template = self.data[index]["land_template"]

        return {'vertices': torch.FloatTensor(vertices), 
                'template': torch.FloatTensor(template), 
                'land_vertices': torch.FloatTensor(land_vertices), 
                'land_template': torch.FloatTensor(land_template)}

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    vertices_path_evoca = args.vertices_path_evoca
    landmarks_path_evoca = args.landmarks_path_evoca

    template_file = args.template_file_voca
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
            
    k=0

    for r, ds, fs in os.walk(vertices_path_evoca):
        for f in tqdm(fs):
            if f.endswith("npy"):
                
                vertices = np.load(os.path.join(vertices_path_evoca, f), allow_pickle=True)
                landmarks = np.load(os.path.join(landmarks_path_evoca, f), allow_pickle=True)
                subject_id = "_".join(f.split("_")[:-3])
                temp = templates[subject_id]
                landmarks_temp = get_landmarks(temp).reshape(-1)
                
                for i in range(vertices.shape[0]):
                    key = f[:-4] + str(i).zfill(3)

                    data[key]["vertices"] = np.reshape(vertices[i], (5023, 3))
                    
                    data[key]["template"] = temp
                    data[key]["land_template"] = landmarks_temp
                    data[key]["land_vertices"] = landmarks[i].reshape(-1)
                    
                
                k += 1
                if k > 1500:
                    break
                        


    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-3])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.mb, shuffle=True, num_workers=8, pin_memory=True)
    valid_data = Dataset(valid_data)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=args.mb, shuffle=True, num_workers=8, pin_memory=True)
    test_data = Dataset(test_data)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=args.mb, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
