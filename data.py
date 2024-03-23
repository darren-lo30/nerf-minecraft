import os
import json
import torchvision
import torch

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

    split_imgs, split_rotations, split_transforms = [], [], []
    for frame in frames:
      img_path = frame['file_path'] + '.png'
      img_path = os.path.normpath(os.path.join(path, img_path))

      img = torchvision.io.read_image(img_path)

      rotation = torch.tensor(frame['rotation'])

      transform_mat = torch.tensor(frame['transform_matrix'])

      split_imgs.append(img)
      split_rotations.append(rotation)
      split_transforms.append(transform_mat)
    
      width = img.shape[1]
      # Distance from camera to img plane
      focal_length = (0.5 * width) / torch.tan(0.5 * camera_angle)
      dataset[split] = (split_imgs, split_rotations, split_transforms, focal_length)

  return dataset

def get_rays(img, focal_length, camera_to_world):
  height = img.shape[0]
  width = img.shape[1]

  # Camera space
  c_ray_dir = torch.cartesian_prod(torch.arange(width), torch.arange(height), torch.tensor([focal_length]))
  print(c_ray_dir)

  c_ray_origin = torch.zeros((3, 1))
  # In camera space, the ray origins is 0, the ray direction are the points
  # Transform both to world space

  # World space
  w_ray_dir = camera_to_world @ c_ray_dir
  w_ray_origin = camera_to_world @ c_ray_origin

  return w_ray_dir, w_ray_origin

  