# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# obj file dataset
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy.lib.format import MAGIC_PREFIX
from PIL import Image
from torch.autograd import Variable
from .ray_utils import *
from torchvision import transforms as T


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


def check_path(path):
    if not os.path.exists(path):
        sys.stderr.write("%s does not exist!\n" % (path))
        sys.exit(-1)


def load_krt(path):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "dist": np.array(dist),
                "extrin": np.array(extrin),
            }

    return cameras

def get_intrin_extrin(base, cam, scale=16):
    cameras = load_krt(base+'/KRT')
    extrin, intrin = cameras[cam]["extrin"], cameras[cam]["intrin"]


    intrin[0,0] /= scale
    intrin[1,1] /= scale
    intrin[0,2] /= scale
    intrin[1,2] /= scale

    return intrin, extrin

def get_vertex(base):
    objfile = base+'/tracked_mesh/E001_Neutral_Eyes_Open'+ '/000102.obj'
    lines = []
    with open(objfile) as obj:
        lines = obj.readlines()
    lines = [line.strip() for line in lines]
    vertex = [line.split(' ')[1:] for line in lines if line.split(' ')[0]=='v']
    vertex = np.array(vertex, dtype=np.float)
    
    return vertex

def get_near_far(rays_o, rays_d, vertex):
    rays_o_np, rays_d_np = rays_o.numpy(), rays_d.numpy()
    distance = (vertex[0,2] - rays_o_np[:,2])/rays_d_np[:,2]
    far, near = np.max(distance), np.min(distance)
    
    return near, far

def get_c2w(extrin):
    R, RT = extrin[:,:3], extrin[:,3]
    T = np.linalg.inv(R) @ RT
    o = -T

    Rc2w = np.linalg.inv(R)
    Tc2w = np.concatenate([np.eye(3), np.reshape(-RT,(3,1))], axis=-1)
    RTc2w = Rc2w @ Tc2w
    return RTc2w

class MultiFaceDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        self.base = 'C:/Users/최준영/Documents/Projects/multiface/download/m--20180227--0000--6795937--GHS'
        krt = load_krt(self.base+'/KRT')
        self.cams = list(krt.keys())
        self.sentnum = 'E001_Neutral_Eyes_Open'
        self.frame = '000102'
        self.photopath = self.base + '/images'
        self.scale = 4
        self.define_transforms()
        self.white_back = False

        self.all_rays = []
        self.all_rgbs = []
        self.cameras = []
        self.cams.remove('400008')
        self.cams.remove('400050')
        for cam in self.cams:
            path = "{}/{}/{}/{}.png".format(self.photopath, self.sentnum, cam, self.frame)
            img = Image.open(path)
            img = img.resize((img.size[0]//self.scale, img.size[1]//self.scale), Image.LANCZOS)

            w, h = img.size[0], img.size[1]
            print('image size', w, h)

            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)

            intrin, extrin = get_intrin_extrin(self.base, cam, scale=self.scale)
            c2w = get_c2w(extrin)
            c2w = torch.FloatTensor(c2w)
            #intrin[2,2] = -1

            #M = intrin @ extrin
            directions = get_ray_directions_face(h, w, intrin[0,0], intrin[0,2], intrin[1,2])
            rays_o, rays_d = get_rays(directions, c2w)
            
            near = -900
            far = -1300

            print('ray_example', rays_o[0]+rays_d[0]*near)
            print(cam)

            self.all_rays += [torch.cat([rays_o, rays_d, 
                                                near*torch.ones_like(rays_o[:, :1]),
                                                far*torch.ones_like(rays_o[:, :1])],
                                                1)]
            self.all_rgbs += [img]
            self.cameras += [cam]*rays_o.shape[0]
        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_rgbs = torch.cat(self.all_rgbs, 0)
    
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):

        if self.split == 'train':
            rays = self.all_rays[idx]
            img = self.all_rgbs[idx]
            cam = self.cameras[idx]

            intrin, extrin = get_intrin_extrin(self.base, cam, scale=self.scale)
            c2w = get_c2w(extrin)
            c2w = torch.FloatTensor(c2w)

            sample = {'rays': rays,
                        'rgbs': img,
                        'c2w': c2w}
            return sample
        
        else:
            cam = self.cameras[idx]
            # image
            #print(self.photopath, self.sentnum, cam, self.frame)
            path = "{}/{}/{}/{}.png".format(self.photopath, self.sentnum, cam, self.frame)
            img = Image.open(path)
            img = img.resize((img.size[0]//self.scale, img.size[1]//self.scale), Image.LANCZOS)
            w, h = img.size[0], img.size[1]

            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)

            intrin, extrin = get_intrin_extrin(self.base, cam, scale=self.scale)
            c2w = get_c2w(extrin)
            c2w = torch.FloatTensor(c2w)

            #M = intrin @ extrin
            directions = get_ray_directions_face(h, w, intrin[0,0], intrin[0,2], intrin[1,2])
            rays_o, rays_d = get_rays(directions, c2w)
            near = -900
            far = -1300

            rays = torch.cat([rays_o, rays_d, 
                                near*torch.ones_like(rays_o[:, :1]),
                                far*torch.ones_like(rays_o[:, :1])],
                                1)

            sample = {'rays': rays,
                        'rgbs': img,
                        'c2w': c2w}
            
            return sample