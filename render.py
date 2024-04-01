from data import get_rays
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import imageio


def render_img(nerf, img_dims, focal_length, camera_transform):
  width, height = img_dims
  img = torch.zeros((3, height, width))

  rays_d, ray_o, _ = get_rays(img, focal_length, camera_transform)
  ray_o = ray_o.expand(rays_d.shape[0], -1)

  dataloader = DataLoader(TensorDataset(rays_d, ray_o), batch_size=4096, shuffle=False)
  colors = []

  for d, o in dataloader:
    d, o = d.to(nerf.device), o.to(nerf.device)
    # HERE
    with torch.no_grad():
      colors.append(nerf.compute_color(o, d))

  colors = torch.cat(colors, dim=0)  # (height * width, 3)
  colors = colors.swapaxes(0, 1)
  colors = colors.reshape((3, height, width))
  colors = torch.clamp(colors, 0, 1)

  return colors

def get_transform_mat(pos, direction):
  # z is up
  # -y is forward
  t = torch.zeros((4, 4), dtype=torch.float32)
  t[:3, 3] = pos
  t[3, 3] = 1

  def normalize(v):
    return torch.nn.functional.normalize(v, p = 2.0, dim = 0)
  
  world_up = torch.tensor([0, 0, 1], dtype=torch.float32)
  forward = normalize(direction)
  right = normalize(torch.cross(world_up, forward))
  up = normalize(torch.cross(right, forward))

  # Change of basis
  t[:3, 0] = right
  t[:3, 1] = -forward # Want basis (0, -1, 0) to change to forward
  t[:3, 2] = up

  return t
  
 
def render_gif(nerf, gif_dims, focal_length, camera_height, camera_radius):
  images = []
  num_frames = 20
  thetas = torch.linspace(0, 2 * torch.pi, num_frames)
  xs = torch.sin(thetas) * camera_radius
  zs = torch.tensor([camera_height]).expand(num_frames)
  ys = torch.cos(thetas) * camera_radius

  for (x, y, z) in zip(xs, ys, zs):
    pos = torch.tensor([x, y, z], dtype=torch.float32)
    # View direction is to origin
    transform = get_transform_mat(pos, -pos)

    img = render_img(nerf, gif_dims, focal_length, transform).cpu()
    img = img.permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8)
    images.append(img)
  imageio.mimsave('./results/test.gif', images)






