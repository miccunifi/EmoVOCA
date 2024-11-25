import numpy as np
import shape_data
import pickle
import trimesh
from sklearn.metrics.pairwise import euclidean_distances
import spiral_utils
import torch
import emovoca_model as emovoca
import os
import shutil
import argparse


def generate_emovoca(args, label_dict, data_dict):
    # Use parsed arguments
    neutral_sequences_path = args.neutral_sequences_path
    audio_path = args.audio_path
    model_path = args.model_path
    conditions_path = args.conditions_path
    coma_templates = args.coma_templates
    intensities = args.intensities
    seq_path = args.seq_path
    label_path = args.label_path
    intensity_path = args.intensity_path
    new_audio_path = args.new_audio_path

    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    latent_size = 256
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]
    device = 'cpu'

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(
        nVal=100,
        reference_mesh_file='/mnt/diskone-second/D2D/template/flame_model/FLAME_sample.ply',
        normalization=False,
        meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    with open('/mnt/diskone-second/D2D/template/template/downsampling_matrices.pkl', 'rb') as fp:
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

    spirals_np[0] = spirals_np[0][:, :-1, :]
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

    tD[0] = tD[0][:, :, 1:]
    tU[0] = tU[0][:, 1:, :]


    model = emovoca.SpiralAutoencoder(filters_enc=filter_sizes_enc,
                                        filters_dec=filter_sizes_dec,
                                        latent_size=latent_size,
                                        sizes=sizes,
                                        spiral_sizes=spiral_sizes,
                                        spirals=tspirals,
                                        D=tD, U=tU, device=device).to(device)


    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])


    with open('/mnt/diskone-first/TH/S2L/vocaset/templates.pkl', 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

    for file in os.listdir(neutral_sequences_path):
        n_seq = np.load(os.path.join(neutral_sequences_path, file))
        n_seq = np.reshape(n_seq, (n_seq.shape[0], 5023, 3))
        actor = file[:24]
        template = templates[actor]
        coma_actor = data_dict[actor]
        coma_template = trimesh.load(os.path.join(coma_templates, 'COMA_' + coma_actor + '.ply'), process=False)
        for condition in os.listdir(os.path.join(conditions_path, 'COMA_' + coma_actor)):
            e_mesh = trimesh.load(os.path.join(conditions_path, 'COMA_' + coma_actor, condition), process=False)
            for intensity in intensities:
                seq = []
                if not os.path.exists(os.path.join(seq_path, file[:-4] + '_' + condition.split('_')[0] + '_' + str(intensity) + '.npy')):
                    for k in range(n_seq.shape[0]):
                        model.eval()
                        with torch.no_grad():
                            mixture, talking, emotional = model.predict(torch.tensor(n_seq[k]-template).unsqueeze(0).to(device).float(), torch.tensor(e_mesh.vertices - coma_template.vertices).unsqueeze(0).to(device).float(), index_talk=2, index_emotional=intensity)
                        mixture = mixture.squeeze(0).cpu().numpy()
                        seq.append(np.array(template + mixture))
                    np.save(os.path.join(seq_path, file[:-4] + '_' + condition.split('_')[0] + '_' + str(intensity) + '.npy'), np.asarray(seq))
                    np.save(os.path.join(label_path, file[:-4] + '_' + condition.split('_')[0] + '_' + str(intensity) + '.npy'), int(label_dict[condition.split('_')[0]]))
                    np.save(os.path.join(intensity_path, file[:-4] + '_' + condition.split('_')[0] + '_' + str(intensity) + '.npy'), intensity)
                    shutil.copy(os.path.join(audio_path, file[:-4] + '.wav'), os.path.join(new_audio_path, file[:-4] + '_' + condition.split('_')[0] + '_' + str(intensity) + '.wav'))

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="EmoVOCA generator")
    parser.add_argument('--neutral_sequences_path', type=str, default='/.vocaset/vertices_npy', help="Path to neutral sequences.")
    parser.add_argument('--audio_path', type=str, default='./vocaset/wav', help="Path to audio files.")
    parser.add_argument('--model_path', type=str, default='/mnt/diskone-first/TH/EmoVOCA_generator/saves', help="Path to pre-trained model.")
    parser.add_argument('--conditions_path', type=str, default='/Conditions', help="Path to conditions.")
    parser.add_argument('--coma_templates', type=str, default='Florence4D', help="Path to Florence COMA templates.")
    parser.add_argument('--intensities', type=int, nargs='+', default=[1, 2, 3], help="List of intensities.")
    parser.add_argument('--seq_path', type=str, default='EmoVOCA/sequences', help="Path to save sequences.")
    parser.add_argument('--label_path', type=str, default='EmoVOCA/label', help="Path to save labels.")
    parser.add_argument('--intensity_path', type=str, default='EmoVOCA/intensity', help="Path to save intensities.")
    parser.add_argument('--new_audio_path', type=str, default='EmoVOCA/wav', help="Path to save audio.")

    args = parser.parse_args()
    
    #Dictionary defining the labels for each expression
    label_dict = {
        "Afraid": "0",
        "Disgust": "1",
        "Irritated1": "2",
        "Sad1": "3",
        "Smile2": "4",
        "Drunk2": "5",
        "Ill": "6",
        "Moody": "7",
        "Pleased": "8",
        "Suspicious": "9",
        "Upset": "10"
        }
    
    #Code to map the Vocaset actor names to the COMA template names
    data_dict = {
        "FaceTalk_170725_00137_TA": "CH01",
        "FaceTalk_170728_03272_TA": "CH02",
        "FaceTalk_170731_00024_TA": "CH03",
        "FaceTalk_170809_00138_TA": "CH04",
        "FaceTalk_170811_03274_TA": "CH05",
        "FaceTalk_170811_03275_TA": "CH06",
        "FaceTalk_170904_00128_TA": "CH07",
        "FaceTalk_170904_03276_TA": "CH08",
        "FaceTalk_170908_03277_TA": "CH09",
        "FaceTalk_170912_03278_TA": "CH10",
        "FaceTalk_170913_03279_TA": "CH11",
        "FaceTalk_170915_00223_TA": "CH12"
        }
    
    generate_emovoca(args, label_dict, data_dict)

if __name__ == "__main__":
    main()