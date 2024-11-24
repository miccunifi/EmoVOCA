import torch.nn as nn
from New_S2L.wav2vec import Wav2Vec2Model
import torch
import numpy as np

def generate_periodic_interpolation(label1, label2, n_frames, device, period=15):
    # Convert labels to PyTorch tensors
    label1 = torch.tensor(label1, dtype=torch.float).cpu()
    label2 = torch.tensor(label2, dtype=torch.float).cpu()
    
    # Calculate interpolation steps
    steps = torch.linspace(0, 1, steps=period)
    
    # Interpolate between the two labels
    interpolated_labels = []
    count = 0
    for period in range(int(n_frames/period)+1):
        for step in steps:
            interpolated_label = (1 - step) * label1 + step * label2
            interpolated_labels.append(interpolated_label)
            count+=1
            if count == n_frames:
                break
        if count == n_frames:
            break
        
    return torch.tensor(np.array(interpolated_labels )).to(device)

class Speech2Land(nn.Module):
    def __init__(self, args):
        super(Speech2Land, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, L*3)
        landmarks: (batch_size, seq_len, L*3)
        """
        #self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.hidden_size = args.feature_dim
        self.drop_prob = 0.2
        self.num_layers = args.num_layers
        self.device = args.device

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.output_size = args.landmarks_dim
        self.audio_mapper = nn.Linear(768, self.hidden_size)

        self.lstm = nn.LSTM(input_size=self.hidden_size*3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.drop_prob)


        self.output_mapper = nn.Linear(self.hidden_size*2, self.output_size)
        nn.init.constant_(self.output_mapper.weight, 0)
        nn.init.constant_(self.output_mapper.bias, 0)
        
        self.label_mapper = nn.Linear(11, self.hidden_size)
        self.intensity_mapper = nn.Linear(3, self.hidden_size)


    def forward(self, audio, landmarks, template, emotion_label, intensity_label):

        frame_num = landmarks.shape[1]

        hidden_states = self.audio_encoder(audio, frame_num=frame_num).last_hidden_state
        
        hidden_states = self.audio_mapper(hidden_states)
        
        label_emb = self.label_mapper(emotion_label).expand(1, landmarks.shape[1], self.hidden_size)
                
        intensity_emb = self.intensity_mapper(intensity_label).expand(1, landmarks.shape[1], self.hidden_size)
        
        input = torch.cat([hidden_states, label_emb, intensity_emb], dim=2)

        displacements_emb, _ = self.lstm(input)

        displacements_pred = self.output_mapper(displacements_emb)

        return displacements_pred + template.unsqueeze(0)
    

    def predict1(self, audio, template, emotion_label, intensity_label1, intensity_label2):

        hidden_states = self.audio_encoder(audio).last_hidden_state
        
        hidden_states = self.audio_mapper(hidden_states)
        
        label_emb = self.label_mapper(emotion_label).expand(1, hidden_states.shape[1], self.hidden_size)
        
        intensity_emb = self.intensity_mapper(generate_periodic_interpolation(intensity_label1, intensity_label2, hidden_states.shape[1], self.device)).unsqueeze(0)
            
        input = torch.cat([hidden_states, label_emb, intensity_emb], dim=2)

        displacements_emb, _ = self.lstm(input)

        displacements_pred = self.output_mapper(displacements_emb)

        return displacements_pred + template.unsqueeze(0)
    
    
    def predict2(self, audio, template, emotion_label, intensity_label):

        hidden_states = self.audio_encoder(audio).last_hidden_state
        
        hidden_states = self.audio_mapper(hidden_states)
        
        label_emb = self.label_mapper(emotion_label).expand(1, hidden_states.shape[1], self.hidden_size)
        
        intensity_emb = self.intensity_mapper(intensity_label).expand(1, hidden_states.shape[1], self.hidden_size)
        
        input = torch.cat([hidden_states, label_emb, intensity_emb], dim=2)

        displacements_emb, _ = self.lstm(input)

        displacements_pred = self.output_mapper(displacements_emb)

        return displacements_pred + template.unsqueeze(0)

        

