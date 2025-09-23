import glob, os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import pandas as pd
def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area
def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv

# Map relevant classes to {0,1,...,19}, and ignored classes to 255
remapper = np.ones(150) * (255)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 22]):
    # 22 is for ceiling
    remapper[x] = i

MATTERPORT_CLASS_REMAP = np.zeros(41)
MATTERPORT_CLASS_REMAP[1] = 1
MATTERPORT_CLASS_REMAP[2] = 2
MATTERPORT_CLASS_REMAP[3] = 3
MATTERPORT_CLASS_REMAP[4] = 4
MATTERPORT_CLASS_REMAP[5] = 5
MATTERPORT_CLASS_REMAP[6] = 6
MATTERPORT_CLASS_REMAP[7] = 7
MATTERPORT_CLASS_REMAP[8] = 8
MATTERPORT_CLASS_REMAP[9] = 9
MATTERPORT_CLASS_REMAP[10] = 10
MATTERPORT_CLASS_REMAP[11] = 11
MATTERPORT_CLASS_REMAP[12] = 12
MATTERPORT_CLASS_REMAP[14] = 13
MATTERPORT_CLASS_REMAP[16] = 14
MATTERPORT_CLASS_REMAP[22] = 21  # DIFFERENCE TO SCANNET!
MATTERPORT_CLASS_REMAP[24] = 15
MATTERPORT_CLASS_REMAP[28] = 16
MATTERPORT_CLASS_REMAP[33] = 17
MATTERPORT_CLASS_REMAP[34] = 18
MATTERPORT_CLASS_REMAP[36] = 19
MATTERPORT_CLASS_REMAP[39] = 20

MATTERPORT_ALLOWED_NYU_CLASSES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 22, 24, 28, 33, 34, 36, 39]


def process_one_scene(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-3]
    region_name = fn.split('/')[-1].split('.')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1
    faces = np.stack(a["face"].data["vertex_indices"], axis=0)
    normal = vertex_normal(coords, faces).astype(np.float32)

    # import pdb;pdb.set_trace()
    category_id = a['face']['category_id']
    category_id[category_id==-1] = 0
    mapped_labels = mapping[category_id]
    mapped_labels[np.logical_not(
            np.isin(mapped_labels, MATTERPORT_ALLOWED_NYU_CLASSES))] = 0

    remapped_labels = MATTERPORT_CLASS_REMAP[mapped_labels].astype(int)

    triangles = a['face']['vertex_indices']
    vertex_labels = np.zeros((coords.shape[0], 22), dtype=np.int32)
    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i],
                            remapped_labels[row_id]] += 1

    vertex_labels = np.argmax(vertex_labels, axis=1)
    vertex_labels[vertex_labels==0] = 256
    vertex_labels -= 1

    torch.save((coords, colors, normal, vertex_labels),
            os.path.join(out_dir,  scene_name+'_' + region_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'train' # 'train' | 'val' | 'test'
out_dir = '/data1/matterport/matterport_3d/{}'.format(split)
matterport_path = '/data1/matterport/v1/scans' # downloaded original matterport data
tsv_file = '/root/code/CUA_O3D/dataset/matterport/category_mapping.tsv'
scene_list = process_txt('/root/code/CUA_O3D/dataset/matterport/scenes_{}.txt'.format(split))
#####################################

os.makedirs(out_dir, exist_ok=True)
category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)

nyu40id_values = category_mapping['nyu40id'].fillna(0).astype(int).to_numpy()
mapping = np.insert(nyu40id_values, 0, 0, axis=0)
# mapping = np.insert(category_mapping[['nyu40id']].to_numpy()
#                         .astype(int).flatten(), 0, 0, axis=0)
files = []
for scene in scene_list:
    files = files + glob.glob(os.path.join(matterport_path, scene, 'region_segmentations', '*.ply'))

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_scene, files)
p.close()
p.join()
