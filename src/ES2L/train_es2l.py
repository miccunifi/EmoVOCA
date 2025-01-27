import os
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
from data_loader import get_dataloaders
from es2l import EmotionalSpeech2Land

class Masked_Velocity_Loss(nn.Module):
    def __init__(self):
        super(Masked_Velocity_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predictions, target):
        
        rec_loss = torch.mean(self.mse(predictions, target))
        
        predictions_ = predictions.squeeze(0)
        target_ = target.squeeze(0)
        
        predictions_ = torch.reshape(predictions_, (predictions_.shape[0], 68, 3))
        target_ = torch.reshape(target_, (target_.shape[0], 68, 3))
        
        mouth_prediction = predictions_[:, 48:, :]
        mouth_target = target_[:, 48:, :]
        
        mouth_rec_loss = torch.mean(self.mse(mouth_prediction, mouth_target))

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + 5 * mouth_rec_loss + 10 * vel_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):

    device = args.device


    es2l = EmotionalSpeech2Land(args).to(args.device)

    print("model parameters: ", count_parameters(es2l))

    dataset = get_dataloaders(args)
       
    optim = torch.optim.Adam(es2l.parameters(), lr=args.lr)
            
    criterion_train = Masked_Velocity_Loss()

    for epoch in range(args.epochs + 1):
        es2l.train()
        tloss = 0
        
        pbar_train = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_train:
            optim.zero_grad()
            audio = sample['audio'].to(device)
            landmarks = sample['landmarks'].to(device)
            template = sample['template_landmarks'].to(device)
            label = sample['label'].to(device)
            intensity = sample['intensity'].to(device)

            landmarks_pred = es2l.forward(audio, landmarks, template, label, intensity)
            loss = criterion_train(landmarks_pred, landmarks) 
            loss.backward()
            optim.step()
            tloss += loss.item()
            pbar_train.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss/(b+1)))
        

        if epoch % 20 == 0 or epoch == args.epochs:
            es2l.eval()
            with torch.no_grad():
                vloss = 0
                pbar_val = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_val:
                    audio = sample['audio'].to(device)
                    landmarks = sample['landmarks'].to(device)
                    template = sample['template_landmarks'].to(device)
                    label = sample['label'].to(device)
                    intensity = sample['intensity'].to(device)
                    landmarks_pred = es2l.forward(audio, landmarks, template, label, intensity)
                    loss = criterion_train(landmarks_pred, landmarks)
                    vloss += loss.item()

                    pbar_val.set_description(
                        "(Epoch {}) TEST MSE:{:.10f}".format((epoch + 1), vloss/(b+1)))
                
               
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': es2l.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(args.result_dir, 'es2l.tar'))
    
        
def main():
    parser = argparse.ArgumentParser(description='Speech2Landmarks')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--landmarks_dim", type=int, default=68*3, help='number of landmarks - 68*3')
    parser.add_argument("--audio_feature_dim", type=int, default=64*3, help='768 for wav2vec')
    parser.add_argument("--feature_dim", type=int, default=32, help='64')
    parser.add_argument("--wav_path", type=str, default="../Dataset/EmoVOCA/wav", help='path of the audio signals')
    parser.add_argument("--landmarks_path", type=str, default="../Dataset/EmoVOCA/landmarks_sequences", help='path of the ground truth')
    parser.add_argument("--intensity_path", type=str, default="../Dataset/EmoVOCA/intensity", help='path of the ground truth')
    parser.add_argument("--label_path", type=str, default="../Dataset/EmoVOCA/label", help='path of the ground truth')
    parser.add_argument("--result_dir", type=str, default="./saves")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--template_file_voca", type=str, default="../Dataset/vocaset/templates.pkl", help='template_file')
    parser.add_argument("--num_layers", type=int, default=7, help='number of S2L layers')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    parser.add_argument("--info", type=str, default="")
    

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
