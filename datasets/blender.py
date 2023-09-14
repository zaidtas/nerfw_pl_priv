import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

from .ray_utils import *

def add_perturbation(img, perturbation, seed, random_occ = True):
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        if random_occ:
            np.random.seed(seed)
            left = np.random.randint(200, 400)
            top = np.random.randint(200, 400)
            for i in range(10):
                np.random.seed(10*seed+i)
                random_color = tuple(np.random.choice(range(256), 3))
                draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                                fill=random_color)
        else:
            draw.rectangle(((200, 200), (400, 400)), fill=(0, 0, 0))
    return img


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800),
                 perturbation=[], random_occ = True, occ_yaw = 0.0, yaw_threshold = 20.0, all_img_occ = False):
        self.root_dir = root_dir
        self.occ_yaw = occ_yaw
        self.yaw_threshold = yaw_threshold
        self.all_img_occ = all_img_occ
        self.add_to_public_data = [3,13,21,33,48,55]
        self.split = split
        self.random_occ = random_occ
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        # import pdb; pdb.set_trace()
        self.define_transforms()

        assert set(perturbation).issubset({"color", "occ"}), \
            'Only "color" and "occ" perturbations are supported!'
        self.perturbation = perturbation
        if self.split == 'train':
            print(f'add {self.perturbation} perturbation!')
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split.split('_')[-1]}.json"), 'r') as f:
            self.meta = json.load(f)
        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w/2
        self.K[1, 2] = h/2

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.K) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            for t, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)
                 # Extract the 3x3 rotation matrix (R) from the 4x4 transformation matrix
                rotation_matrix = pose[:3, :3]
                
                # Calculate Euler angles (yaw, pitch, roll) from the rotation matrix
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
                # roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                
                # Convert angles from radians to degrees
                yaw_deg = np.degrees(yaw)
                pitch_deg = np.degrees(pitch)
                # roll_deg = np.degrees(roll)


                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)

                # if self.perturbation == [] #and np.abs(yaw_deg-self.occ_yaw)<self.yaw_threshold and np.abs(pitch_deg-self.occ_yaw)<self.yaw_threshold and t not in self.add_to_public_data:
                #     #skip the image and print the image path
                #     print('skipping image:', image_path)
                #     continue

                if not self.perturbation == []:
                    if self.random_occ:
                        if t != 0:                        
                            img = add_perturbation(img, self.perturbation, t,random_occ=self.random_occ)
                    else:
                        if (np.abs(yaw_deg-self.occ_yaw)<self.yaw_threshold and np.abs(pitch_deg-self.occ_yaw)<self.yaw_threshold and t not in self.add_to_public_data) or self.all_img_occ:
                            # print('image:', image_path)
                            img = add_perturbation(img, self.perturbation, t,random_occ=self.random_occ)


                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            # self.img_id = t

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'img_id': idx}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

             # Extract the 3x3 rotation matrix (R) from the 4x4 transformation matrix
            rotation_matrix = c2w[:3, :3]
            
            # Calculate Euler angles (yaw, pitch, roll) from the rotation matrix
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
            # roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            
            # Convert angles from radians to degrees
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            # roll_deg = np.degrees(roll)
            t = 0 # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            
            if not self.perturbation == []:
                if self.random_occ:
                    if self.split == 'test_train' and idx != 0:
                        t = idx
                        img = add_perturbation(img, self.perturbation, idx)

                else:
                    if (np.abs(yaw_deg-self.occ_yaw)<self.yaw_threshold and np.abs(pitch_deg-self.occ_yaw)<self.yaw_threshold) or self.all_img_occ :
                        t = idx
                        img = add_perturbation(img, self.perturbation, t,random_occ=self.random_occ)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

            if self.split == 'test_train' and self.perturbation:
                 # append the original (unperturbed) image
                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, H, W)
                valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask

        return sample