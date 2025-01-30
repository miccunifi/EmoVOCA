import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import ES2D.es2d as models
import ES2D.spiral_utils as spiral_utils
import ES2D.shape_data as shape_data
import argparse
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import torch
import Get_landmarks
from ES2L.es2l import EmotionalSpeech2Land
from transformers import Wav2Vec2Processor
import time
import cv2
import tempfile
import numpy as np
from subprocess import call
from psbody.mesh import Mesh
import pyrender
import trimesh
import glob
import librosa

def integer_to_one_hot_encoding(integer, num_classes):
    if integer < 0 or integer >= num_classes:
        raise ValueError("Integer value is out of range for one-hot encoding with the specified number of classes.")
    encoding = [0] * num_classes
    encoding[integer] = 1
    return encoding


def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):

    background_black = True
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    if background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])  # [0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2],
                               bg_color=[255, 255, 255])  # [0, 0, 0] black,[255, 255, 255] white

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, save_path_images, out_fname, fps, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for i_frame in range(num_frames - 2):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
        cv2.imwrite(save_path_images + '/' + str(i_frame).zfill(3) + '.png', img[:, 100:-100])
        i = i + 1
    writer.release()

    video_fname = os.path.join(out_path, out_fname)
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)

def generate_mesh_video(out_path, out_fname, meshes_path_fname, save_path_images, fps, audio_fname, template):

    sequence_fnames = sorted(glob.glob(os.path.join(meshes_path_fname, '*.ply*')))

    audio_fname = audio_fname


    uv_template_fname = template
    sequence_vertices = []
    f = None

    for frame_idx, mesh_fname in enumerate(sequence_fnames):
        frame = Mesh(filename=mesh_fname)
        sequence_vertices.append(frame.v)
        if f is None:
            f = frame.f

    template = Mesh(sequence_vertices[0], f)
    sequence_vertices = np.stack(sequence_vertices)
    render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, save_path_images, out_fname, fps, uv_template_fname=uv_template_fname, texture_img_fname='')


def generate_landmarks(args, label, intensity, model_path, audio_path, template_file):

    model = EmotionalSpeech2Land(args)  
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    model = model.to(torch.device(args.device))
    model.eval()

    speech_array, sampling_rate = librosa.load(os.path.join(audio_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    actor_vertices = templates[args.template_name]
    
    label = torch.FloatTensor(integer_to_one_hot_encoding(int(label), 11)).to(device=args.device)
    intensity = torch.FloatTensor(integer_to_one_hot_encoding(int(intensity), 3)).to(device=args.device)

    actor_landmarks = Get_landmarks.get_landmarks(actor_vertices)
    actor = actor_landmarks.reshape((-1))
    actor = np.reshape(actor, (-1, actor.shape[0]))
    actor = torch.FloatTensor(actor).to(device=args.device)

    prediction = model.predict(audio_feature, actor, label, intensity)

    prediction = prediction.squeeze()  # (seq_len, V*3)
    landmarks = prediction.detach().cpu().numpy()
    landmarks = np.reshape(landmarks, (landmarks.shape[0], 68, 3))

    return landmarks, actor_landmarks, actor_vertices

def generate_meshes_from_landmarks(landmarks, template_lands, template, reference_mesh_path, prediction_path, args):


    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    nz = 16
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    nbr_landmarks = 68
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]

    meshpackage = 'trimesh'
    
    temp_faces = trimesh.load(reference_mesh_path, process=False).faces

    shapedata = shape_data.ShapeData(nVal=100,
                                     test_file='/test.npy',
                                     reference_mesh_file=reference_mesh_path,
                                     normalization=False,
                                     meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    with open(
            './template/template/downsampling_matrices.pkl',
            'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
         range(len(M_verts_faces))]

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
                                                                      reference_points=reference_points,
                                                                      dilation=dilation, random=False,
                                                                      meshpackage='trimesh',
                                                                      counter_clockwise=True)

    sizes = [x.vertices.shape[0] for x in M]

    device = torch.device(args.device)

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



    model = models.SpiralDecoder(filters_enc=filter_sizes_enc,
                                     filters_dec=filter_sizes_dec,
                                     latent_size=nz,
                                     sizes=sizes,
                                     nbr_landmarks=nbr_landmarks,
                                     spiral_sizes=spiral_sizes,
                                     spirals=tspirals,
                                     D=tD, U=tU, device=device).to(device)

    checkpoint = torch.load(args.S2D, map_location=device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    
    
    for k in range(landmarks.shape[0]):
        _, pred = model(torch.tensor(landmarks[k] - template_lands).reshape(-1).unsqueeze(0).float().to(device), torch.tensor(template).to(device)) 
        mesh = trimesh.Trimesh(vertices=pred.squeeze(0).detach().cpu().numpy(), faces=temp_faces)
        mesh.export(os.path.join(prediction_path, 'tst' + str(k).zfill(3) + '.ply'))
    

    print('Done')

def main():
    parser = argparse.ArgumentParser(description='S2L+S2D: Speech-Driven 3D Talking heads')
    parser.add_argument("--landmarks_dim", type=int, default=68 * 3, help='number of landmarks - 68*3')
    parser.add_argument("--audio_feature_dim", type=int, default=64*3, help='768 for wav2vec')
    parser.add_argument("--feature_dim", type=int, default=32, help='64 for vocaset')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_layers", type=int, default=7, help='number of S2L layers')
    parser.add_argument("--S2L", type=str, default='./ES2L/saves/es2l.tar', help='path to the S2L model')
    parser.add_argument("--S2D", type=str, default='./ES2D/saves/s2d.tar', help='path to the S2D model')
    parser.add_argument("--template_file", type=str, default="./Dataset/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--template_name", type=str, default="FaceTalk_170809_00138_TA", help='face to animate')
    parser.add_argument("--audio_path", type=str, default='./Example/photo.wav', help='audio to animate')
    parser.add_argument("--save_path", type=str, default='prediction', help='path for results')
    parser.add_argument("--flame_template", type=str, default="./Dataset/vocaset/flame_model/FLAME_sample.ply", help='template_path')
    parser.add_argument("--video_name", type=str, default="Example_Happy_photo.mp4", help='name of the rendered video')
    parser.add_argument("--fps", type=int, default=60, help='frames per second')
    parser.add_argument("--Emotion", type=str, default='Happy', help='Expression to animate')
    parser.add_argument("--Intensity", type=str, default='Mid', help='Intensity to animate')

    
    args = parser.parse_args()
    test_audio_path = args.audio_path
    save_path = args.save_path
    

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'Meshes'))
        os.mkdir(save_path + '/Images')

    model_path = args.S2L
    audio_path = test_audio_path

    actors_file = args.template_file

    template_path = args.flame_template
    prediction_path = os.path.join(save_path, 'Meshes')
    save_path_images = save_path + '/Images'
    
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
    
    intensity_dict = {
        "Low": "0",
        "Mid": "1",
        "High": "2"
        }
    
    label = label_dict[args.Emotion]
    intensity = intensity_dict[args.Intensity]
    
    print('Landmarks generation')
    start = time.time()
    landmarks, actor_lands, actor_verts = generate_landmarks(args, label, intensity, model_path, audio_path, actors_file)

    print('Meshes Generation')

    generate_meshes_from_landmarks(landmarks, actor_lands, actor_verts, args.flame_template, prediction_path, args)
    end = time.time()

    print(str(end - start) + ' Seconds')

    save_video_path = save_path

    print('Video Generation')
    generate_mesh_video(save_video_path,
                        args.video_name,
                        prediction_path,
                        save_path_images,
                        args.fps,
                        audio_path,
                        template_path)
    print('done')

if __name__ == '__main__':
    main()
