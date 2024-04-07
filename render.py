from data import get_rays
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import imageio
import matplotlib.pyplot as plt
import mcschematic
from utils import sample_bins_uniform


def render_img(nerf, img_dims, focal_length, camera_transform, N_points = 64):
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
      batch_size = d.shape[0]
      t = sample_bins_uniform(batch_size, N_points, nerf.t_near, nerf.t_far).to(nerf.device)  # (batch_size, N)
      c, _ = nerf.compute_color(nerf.fine_model, t, o, d)
      colors.append(c)

  colors = torch.cat(colors, dim=0)  # (height * width, 3)
  colors = colors.swapaxes(0, 1)
  colors = colors.reshape((3, height, width))
  colors = torch.clamp(colors, 0, 1)

  return colors


def get_transform_mat(pos, direction):
  # Changes coordinate spaces from camera to world space
  # World space  ~ forward (+y), right (+x), up (+z)
  # Camera space ~ forward (-z), right (+x), up (+y)
  change_coordinate_space = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)

  direction = change_coordinate_space[:3, :3] @ direction
  pos = change_coordinate_space @ torch.cat([pos, torch.tensor([1.0])], dim=-1)
  pos = pos[:3]

  t = torch.zeros((4, 4), dtype=torch.float32)
  t[:3, 3] = pos
  t[3, 3] = 1

  def normalize(v):
    return torch.nn.functional.normalize(v, p=2.0, dim=0)

  world_up = torch.tensor([0, 0, 1], dtype=torch.float32)
  forward = normalize(direction)
  right = normalize(torch.cross(forward, world_up))
  up = normalize(torch.cross(right, forward))

  # Change of basis
  t[:3, 0] = right
  t[:3, 1] = forward
  t[:3, 2] = up

  t = t @ change_coordinate_space

  return t


def render_gif(nerf, gif_dims, focal_length, camera_height, camera_radius):
  images = []
  num_frames = 20
  thetas = torch.linspace(0, 2 * torch.pi, num_frames)
  xs = torch.sin(thetas) * camera_radius
  ys = torch.tensor([camera_height]).expand(num_frames)
  zs = torch.cos(thetas) * camera_radius

  for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
    print(f"Rendering frame {i}")
    pos = torch.tensor([x, y, z], dtype=torch.float32)
    # View direction is to origin
    transform = get_transform_mat(pos, -pos)

    img = render_img(nerf, gif_dims, focal_length, transform).cpu()
    img = img.permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8)
    images.append(img)
  imageio.mimsave("./results/test.gif", images)


# # Render a 3D mesh
# def render_mesh(nerf, world_dims, sample_density = 50, density_threshold = 0.5):
#   grid_points = torch.stack([torch.linspace(-dim, dim, sample_density) for dim in world_dims], dim = -1, dtype=torch.float32)
#   print(grid_points.shape)
#   dirs = torch.zeros_like(grid_points)


# Render a 3D voxel mesh (cubes)
def generate_voxel_map(nerf, world_dims, sample_density=50, density_threshold=0.5):
  x, y, z = torch.meshgrid([torch.linspace(-dim, dim, sample_density) for dim in world_dims], indexing="ij")

  grid_points = torch.stack([x, y, z])
  grid_points = grid_points.permute(1, 2, 3, 0)
  world_shape = grid_points.shape[:3]
  grid_points = grid_points.flatten(start_dim=0, end_dim=2)

  all_densitys = []
  grids_data = DataLoader(grid_points, shuffle=False, batch_size=4096)
  for grid_pt in grids_data:
    null_dirs = torch.zeros_like(grid_pt)

    grid_pt = grid_pt.to(nerf.device)
    null_dirs = null_dirs.to(nerf.device)
    
    _, densitys = nerf.fine_model(grid_pt, null_dirs)
    all_densitys.append(densitys.detach().cpu())

  all_densitys = torch.cat(all_densitys, dim = 0)
  all_densitys = all_densitys.reshape(world_shape)

  voxel_map = all_densitys > density_threshold

  return voxel_map

def render_voxels(voxel_map, out_file):
  voxel_map = voxel_map.cpu().numpy()

  ax = plt.figure().add_subplot(projection='3d')
  ax.voxels(voxel_map, edgecolor='k')

  plt.savefig(out_file)

# Generate a minecraft world with blocks
def generate_mc_schematic(voxel_map, save_folder, schem_name):
  schem = mcschematic.MCSchematic()

  for x in range(voxel_map.shape[0]):
    for y in range(voxel_map.shape[1]):
      for z in range(voxel_map.shape[2]):
        # y in mc is up but z in nerf world space
        if(voxel_map[x, y, z]):
          schem.setBlock((x, z, y), "minecraft:stone")

  schem.save(save_folder, schem_name, mcschematic.Version.JE_1_16_5)
