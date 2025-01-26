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
from s2d_spiral import SpiralDecoder
from sklearn.metrics.pairwise import euclidean_distances
import shape_data
import spiral_utils

class MVE(nn.Module):
    def __init__(self):
        super(MVE, self).__init__()

    def forward(self, target, predictions):
        mve = torch.square(target - predictions)
        mve = torch.sum(mve, dim=2)
        mve, _ = torch.max(mve, dim=1)
        mve = torch.mean(mve)
        return mve

def loss_weighted(disp_pred, disp, face_pred, face, loss_weights):
    L = (torch.matmul(loss_weights, torch.abs(face_pred - face))).mean() + 0.1 * (torch.abs(
        disp_pred - disp)).mean()
    return L

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):

    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    nz = 16
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    nbr_landmarks = 68
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                          test_file=args.root_dir + '/test.npy',
                          reference_mesh_file=args.reference_mesh_file,
                          normalization=False,
                          meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3


    with open('../template/template/downsampling_matrices.pkl', 'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in range(len(M_verts_faces))]

    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    for i in range(len(ds_factors)):
        dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    Adj, Trigs = spiral_utils.get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage='trimesh')

    spirals_np, spiral_sizes, spirals = spiral_utils.generate_spirals(step_sizes,
                                                            M, Adj, Trigs,
                                                            reference_points = reference_points,
                                                            dilation = dilation, random = False,
                                                            meshpackage = 'trimesh',
                                                            counter_clockwise = True)
    sizes = [x.vertices.shape[0] for x in M]


    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)


    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]



    dataset = get_dataloaders(args)


    s2d = SpiralDecoder(filters_enc=filter_sizes_enc,
                                      filters_dec=filter_sizes_dec,
                                      latent_size=nz,
                                      sizes=sizes,
                                      nbr_landmarks=nbr_landmarks,
                                      spiral_sizes=spiral_sizes,
                                      spirals=tspirals,
                                      D=tD, U=tU, device=device).to(device)
    print("model parameters: ", count_parameters(s2d))

    loss_weights = np.load('../template/template/Normalized_d_weights.npy', allow_pickle=True)
    loss_weights = torch.from_numpy(loss_weights).float().to(device)[:-1]
    
    optim = torch.optim.Adam(s2d.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs + 1):
        s2d.train()
        tloss = 0
        
        pbar_train = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_train:
            optim.zero_grad()
            vertices = sample['vertices'].to(device)
            template = sample['template'].to(device)
            vertices_land = sample['land_vertices'].to(device)
            template_land = sample['land_template'].to(device)

            disp_pred, vertices_pred = s2d.forward(vertices_land - template_land, template)
            loss = loss_weighted(disp_pred, vertices-template, vertices_pred, vertices, loss_weights) 

            loss.backward()
            optim.step()
            
            tloss += loss.item()
            pbar_train.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss/(b+1)))
        

        if epoch % 20 == 0 or epoch == args.epochs:
            s2d.eval()
            with torch.no_grad():
                vloss = 0
                pbar_val = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_val:
                    vertices = sample['vertices'].to(device)
                    template = sample['template'].to(device)
                    vertices_land = sample['land_vertices'].to(device)
                    template_land = sample['land_template'].to(device)

                    disp_pred, vertices_pred = s2d.forward(vertices_land - template_land, template)
                    loss = loss_weighted(disp_pred, vertices-template, vertices_pred, vertices, loss_weights) 
                    vloss += loss.item()

                    pbar_val.set_description(
                        "(Epoch {}) VAL loss:{:.10f}".format((epoch + 1), vloss/(b+1)))
                
               
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': s2d.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(args.result_dir, 'es2l.tar'))
    
        
def main():
    parser = argparse.ArgumentParser(description='Spriral Convolution D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--mb", type=int, default=64, help='number of epochs')
    parser.add_argument("--result_dir", type=str, default="./saves")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reference_mesh_file", type=str, default='../template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--root_dir", type=str, default='')
    parser.add_argument("--voca_transform", type=str, default='../template/template/transform.pkl')
    parser.add_argument("--vertices_path_evoca", type=str, default="../Dataset/EmoVOCA/sequences", help='path of the ground truth')
    parser.add_argument("--landmarks_path_evoca", type=str, default="../Dataset/EmoVOCA/landmarks_sequences", help='path of the ground truth')
    parser.add_argument("--template_file_voca", type=str, default="../Dataset/vocaset/templates.pkl", help='template_file')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
