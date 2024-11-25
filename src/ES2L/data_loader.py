import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import Get_landmarks as Get_landmarks



def integer_to_one_hot_encoding(integer, num_classes):
    if integer < 0 or integer >= num_classes:
        raise ValueError("Integer value is out of range for one-hot encoding with the specified number of classes.")
    encoding = [0] * num_classes
    encoding[integer] = 1
    return encoding


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        landmarks = self.data[index]["landmarks"]
        template = self.data[index]["template"]
        template_landmarks = self.data[index]["template_landmarks"]
        label = self.data[index]["label"]
        intensity = self.data[index]["intensity"]

        return {'audio': torch.FloatTensor(audio), 
                'landmarks': torch.FloatTensor(landmarks), 
                'template_landmarks': torch.FloatTensor(template_landmarks), 
                'template': torch.FloatTensor(template), 
                'label': torch.FloatTensor(label), 
                'intensity': torch.FloatTensor(intensity), 
                'filename': file_name}

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = args.wav_path
    landmarks_path = args.landmarks_path
    label_path = args.label_path
    intensity_path = args.intensity_path
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = args.template_file_voca
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    j = 0
    for r, ds, fs in os.walk(landmarks_path):
        for f in tqdm(fs):
            if f.endswith("npy"): 
                f = f.replace("npy", "wav")
                wav_path = os.path.join(audio_path, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                f = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-3])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["label"] = integer_to_one_hot_encoding(np.load(os.path.join(label_path, f)), 11)
                data[key]["intensity"] = integer_to_one_hot_encoding(np.load(os.path.join(intensity_path, f)) - 1, 3)
                data[key]["template"] = temp
                landmarks_temp = Get_landmarks.get_landmarks(temp)
                data[key]["template_landmarks"] = landmarks_temp.reshape((-1))
                landmarks_path_ = os.path.join(landmarks_path, f.replace("wav", "npy"))
                if not os.path.exists(landmarks_path_):
                    del data[key]
                else:
                    landmarks = np.load(landmarks_path_, allow_pickle=True)
                    if landmarks.shape[1] == 5023:
                        del data[key]
                    else:
                        landmarks = np.reshape(landmarks, (landmarks.shape[0], 204))
                        data[key]["landmarks"] = landmarks
                
                #j+=1
                #if j==5000:
                #  break


    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    #splits = {'train': range(1, 41), 'val': range(1, 41), 'test': range(1, 41)}

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
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
