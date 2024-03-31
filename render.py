from data import get_rays
import torch
from torch.utils.data import TensorDataset, DataLoader

def render_img(nerf, img_dims, focal_length, camera_transform):
  width, height = img_dims
  img = torch.zeros((3, height, width))

  rays_d, ray_o, _ = get_rays(img, focal_length, camera_transform)
  ray_o = ray_o.expand(rays_d.shape[0], -1)

  dataloader = DataLoader(TensorDataset(rays_d, ray_o), batch_size = 4096, shuffle=False)
  colors = []

  for d, o in dataloader:
    d, o = d.to(nerf.device), o.to(nerf.device)
    # HERE
    with torch.no_grad():
      colors.append(nerf.compute_color(o, d))

  colors = torch.cat(colors, dim = 0) # (height * width, 3)
  colors = colors.swapaxes(0, 1)
  colors = colors.reshape((3, height, width))
  colors = torch.clamp(colors, 0, 1)

  return colors
  

  
