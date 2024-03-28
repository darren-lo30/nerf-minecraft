import os
import json
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from random import randrange

class NERFData():
  def __init__(self, imgs, transforms, focal_length):
    self.imgs = imgs
    self.transforms = transforms
    self.focal_length = focal_length

  def sample_rand_img_rays(self, num_rays):
    idx = randrange(len(self.imgs))
    img, transform = self.imgs[idx], self.transforms[idx]

    ray_d, ray_o, colors  = get_rays(img, self.focal_length, transform)
    
    ray_indices = np.random.choice(ray_d.shape[0], num_rays, replace=False)
    sampled_ray_d = ray_d[ray_indices]
    sampled_colors = colors[ray_indices]
    ray_o = ray_o.expand(num_rays, -1)

    return sampled_ray_d, ray_o, sampled_colors
  
def load_data(path):
  splits = ['train', 'test', 'val']
  files = {}
  dataset = {}

  for split in splits:
    with open(os.path.join(path, f'transforms_{split}.json'), 'r') as f:
      files[split] = json.load(f)

  for split in splits:
    file = files[split]
    # FOV of camera (in radians)
    camera_angle = file['camera_angle_x']
    frames = file['frames']

    split_imgs, split_transforms = [], []
    for frame in frames:
      img_path = frame['file_path'] + '.png'
      img_path = os.path.normpath(os.path.join(path, img_path))

      img = torchvision.io.read_image(img_path)

      transform_mat = torch.tensor(frame['transform_matrix'])

      split_imgs.append(img)
      # Theses are the transforms of the camera in world space
      # Equivalent to the camera2world matrix / inverse of view matrix
      split_transforms.append(transform_mat)
    
      width = img.shape[1]
      # Distance from camera to img plane
      focal_length = (0.5 * width) / np.tan(0.5 * camera_angle)
      dataset[split] = NERFData(split_imgs, split_transforms, focal_length)

  return dataset

def get_rays(img, focal_length, camera_to_world):
  height = img.shape[1]
  width = img.shape[2]

  # Camera space
  x, y = torch.meshgrid(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32), indexing='xy')
  # Recenter and rescale
  x = (x - width / 2) / focal_length
  y = (y - height / 2) / focal_length
  c_ray_dir = torch.stack([x, y, -torch.ones_like(x)], dim=-1).flatten(start_dim=0, end_dim=1)
  colors = torch.flatten(img[:3, :, :], start_dim=1) / 255.
  colors = colors.swapaxes(0, 1)

  c_ray_origin = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
  # In camera space, the ray origins is 0, the ray direction are the points
  # Transform both to world space

  # World space
  w_ray_dir = camera_to_world[:3, :3] @ torch.transpose(c_ray_dir, 0, 1)
  w_ray_dir = torch.transpose(w_ray_dir, 0, 1)
  w_ray_origin = camera_to_world @ c_ray_origin
  w_ray_origin = w_ray_origin[:3]  # Discard 4th coordinate

  return w_ray_dir, w_ray_origin, colors  

class NERFRayDataset(Dataset):
  def __init__(self, img, transform, focal_length):
    self.rays_d, self.rays_o, self.colors = get_rays(img, focal_length, transform)

  def __len__(self):
    return len(self.rays_d)
  
  def __getitem__(self, index):
    return self.rays_d[index], self.rays_o, self.colors[index]
