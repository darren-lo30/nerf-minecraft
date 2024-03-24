import os
import json
import torchvision
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

      rotation = torch.tensor(frame['rotation'])

      transform_mat = torch.tensor(frame['transform_matrix'])

      split_imgs.append(img)
      # Theses are the transforms of the camera in world space
      # Equivalent to the camera2world matrix / inverse of view matrix
      split_transforms.append(transform_mat)
    
      width = img.shape[1]
      # Distance from camera to img plane
      focal_length = (0.5 * width) / np.tan(0.5 * camera_angle)
      dataset[split] = (split_imgs, split_transforms, focal_length)

  return dataset

def get_rays(img, focal_length, camera_to_world):
  height = img.shape[0]
  width = img.shape[1]

  # Camera space
  c_ray_dir = torch.cartesian_prod(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32), torch.tensor([focal_length], dtype=torch.float32))
  colors = img[c_ray_dir[:, 1].to(dtype=torch.int), c_ray_dir[:, 0].to(dtype=torch.int)]

  c_ray_origin = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
  # In camera space, the ray origins is 0, the ray direction are the points
  # Transform both to world space

  # World space
  w_ray_dir = camera_to_world[:3, :3] @ torch.transpose(c_ray_dir, 0, 1)
  w_ray_dir = torch.transpose(w_ray_dir, 0, 1)
  w_ray_origin = camera_to_world @ c_ray_origin
  w_ray_origin = w_ray_origin[:3]  # Discard 4th coordinate

  return w_ray_dir, w_ray_origin, colors

class NERFDataset(Dataset):
  def __init__(self, imgs, transforms, focal_length):
    self.rays = []
    for (img, transform) in zip(imgs, transforms):
      rays_d, rays_o, colors = get_rays(img, focal_length, transform)
      for ray_d in rays_d:
        self.rays.append((ray_d, rays_o, colors))

  def __len__(self):
    return len(self.rays)
  
  def __getitem__(self, index):
    return self.rays[index]

def get_dataloaders(path, batch_size = 256):
  datasets = load_data(path)
  train_data = DataLoader(NERFDataset(*datasets['train']), batch_size=batch_size, shuffle=True)
  val_data = DataLoader(NERFDataset(*datasets['val']), batch_size=batch_size, shuffle=False)
  test_data = DataLoader(NERFDataset(*datasets['test']), batch_size=batch_size, shuffle=False)

  return train_data, val_data, test_data