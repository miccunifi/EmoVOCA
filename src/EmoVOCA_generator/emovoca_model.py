import torch
import torch.nn as nn

class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True, device=None):
        super(SpiralConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c * spiral_size, out_c, bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        spirals_index = spiral_adj.view(bsize * num_pts * spiral_size) 
        batch_index = torch.arange(bsize, device=self.device).view(-1, 1).repeat([1, num_pts * spiral_size]).view(-1).long() 
        spirals = x[batch_index, spirals_index, :].view(bsize * num_pts, spiral_size * feats)

        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=self.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat


class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, device,
                 activation='elu'):
        super(SpiralAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation
        self.weights = 0.5

        self.conv_talk = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes) - 1):
            if filters_enc[1][i]:
                self.conv_talk.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv_talk.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i + 1],
                                        activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i + 1]

        self.conv_talk = nn.ModuleList(self.conv_talk)
        
        self.conv_emo = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes) - 1):
            if filters_enc[1][i]:
                self.conv_emo.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv_emo.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i + 1],
                                        activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i + 1]

        self.conv_emo = nn.ModuleList(self.conv_emo)

        self.fc_latent_enc_talk = nn.Linear((sizes[-1] + 1) * input_size, latent_size)
        self.fc_latent_enc_emo = nn.Linear((sizes[-1] + 1) * input_size, latent_size)

        self.fc_latent_dec = nn.Linear(latent_size * 2, (sizes[-1] + 1) * filters_dec[0][0])

        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes) - 1):
            if i != len(spiral_sizes) - 2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i + 1]

                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i + 1]
            else:
                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i + 1]
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[1][i + 1]
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i + 1]

        self.dconv = nn.ModuleList(self.dconv)

    def encode_talking(self, x_talking):
        bsize = x_talking.size(0)
        S = self.spirals
        D = self.D

        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x_talking = self.conv_talk[j](x_talking, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x_talking = self.conv_talk[j](x_talking, S[i].repeat(bsize, 1, 1))
                j += 1
            x_talking = torch.matmul(D[i], x_talking)
        x_talking = x_talking.view(bsize, -1)
        x_talking = self.fc_latent_enc_talk(x_talking)

        return x_talking
    
    def encode_emotional(self, x_emotional):
        bsize = x_emotional.size(0)
        S = self.spirals
        D = self.D
        k = 0
        for i in range(len(self.spiral_sizes) - 1):
            x_emotional = self.conv_emo[k](x_emotional, S[i].repeat(bsize, 1, 1))
            k += 1
            if self.filters_enc[1][i]:
                x_emotional = self.conv_emo[k](x_emotional, S[i].repeat(bsize, 1, 1))
                k += 1
            x_emotional = torch.matmul(D[i], x_emotional)
        x_emotional = x_emotional.view(bsize, -1)
        
        return self.fc_latent_enc_emo(x_emotional)
            

    def decode(self, z1, z2):
        bsize = z1.size(0)
        S = self.spirals
        U = self.U
        z = torch.cat((z1, z2), dim=1)
        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1] + 1, -1)
        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = torch.matmul(U[-1 - i], x)
            x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))

            j += 1
            if self.filters_dec[1][i + 1]:
                x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
                j += 1
        return x
    
    def forward_talking(self, x_talking, template):
        z_talking = self.encode_talking(x_talking)
        x_talking = self.decode(z_talking, z_talking)
        return x_talking, x_talking + template

    def forward_emotional(self, x_emotional, template):
        z_emotional = self.encode_emotional(x_emotional)
        x_emotional = self.decode(z_emotional, z_emotional)
        return x_emotional, x_emotional + template

    def predict(self, x_talking, x_emotional, index_talk=2, index_emotional=2):
        z_emotional = self.encode_emotional(x_emotional)
        z_talking = self.encode_talking(x_talking)
        mixture = self.decode(z_talking * index_talk, z_emotional * index_emotional)
        emotional = self.decode(z_emotional, z_emotional)
        talking = self.decode(z_talking, z_talking)
        return mixture, talking, emotional
    
    def predict_single_encoder(self, x_talking, x_emotional, index_talk=2, index_emotional=2):
        z_emotional = self.encode_talking(x_emotional)
        z_talking = self.encode_talking(x_talking)
        mixture = self.decode(z_talking * index_talk, z_emotional * index_emotional)
        emotional = self.decode(z_emotional, z_emotional)
        talking = self.decode(z_talking, z_talking)
        return mixture, talking, emotional
    
    def get_talking_features(self, x_talking):
        z_talking = self.encode_talking(x_talking)
        return z_talking
    
    def get_emotional_features(self, x_emotional):
        z_emotional = self.encode_emotional(x_emotional)
        return z_emotional