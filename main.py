from simple_nerf import NERF
from data import load_data
from utils import get_device
from render import generate_voxel_map, generate_mc_schematic, generate_mesh, render_gif
import argparse
import sys

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="NERF", description="An implementation of Neural Radiance Fields")

  subparsers = parser.add_subparsers(help="mode", dest="cmd")

  parser.add_argument("train", action="store_true", default=True)
  parser.add_argument("load", action="store_true")

  is_train = "--train" in sys.argv
  is_load = "--load" in sys.argv and not is_train

  train_parser = subparsers.add_parser("train")
  train_parser.add_argument("--data", type=str, required=True)
  train_parser.add_argument("--scale", type=float, default=1.0)
  train_parser.add_argument("--epochs", type=int, default=5000)
  train_parser.add_argument("--save", type=str, default="")
  train_parser.add_argument("--use_fine", type=bool, default=False)

  load_parser = subparsers.add_parser("load")
  load_parser.add_argument("--model", type=str, required=True)
  load_parser.add_argument("--schem_dir", type=str, default="")
  load_parser.add_argument("--mesh_dir", type=str, default="")
  load_parser.add_argument("--gif_dir", type=str, default="")

  load_parser.add_argument("--gif_dims", type=tuple, default=(800, 800))
  load_parser.add_argument("--gif_focal", type=int, default=1111)

  args = parser.parse_args()


  if args.cmd == "train":
    nerf = NERF(device=get_device(), use_fine=args.use_fine)
    data = load_data(args.data, scale_ratio=args.scale)
    nerf.train(args.epochs, data["train"])
    if args.save != "":
      nerf.save(args.save)
  else:
    nerf = NERF(device=get_device())
    nerf.load(args.model)
    if args.schem_dir != "":
      voxel_map = generate_voxel_map(nerf, (1, 1, 1))
      generate_mc_schematic(voxel_map, args.schem_dir, 'my-schem')
    
    if args.mesh_dir != "":
      generate_mesh(nerf, (1, 1, 1), save_path=args.mesh_dir)

    if args.gif_dir != "":
      render_gif(nerf, args.gif_dims, args.gif_focal, 3, 4, args.gif_dir)