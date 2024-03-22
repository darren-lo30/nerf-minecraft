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

def get_rays(img, transform, focal_length):
  height = img.shape[0]
  width = img.shape[1]

  points = torch.tensor(torch.meshgrid(torch.arange(width), torch.arange(height)))
  