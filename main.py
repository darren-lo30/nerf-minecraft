from simple_nerf import SimpleNERFModel
from data import load_data
from utils import get_device
from render import generate_voxel_map, render_voxels, generate_mc_schematic
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

  load_parser = subparsers.add_parser("load")
  load_parser.add_argument("--model", type=str, required=True)

  args = parser.parse_args()

  nerf = SimpleNERFModel(device=get_device())

  if args.cmd == "train":
    data = load_data(args.data, scale_ratio=args.scale)
    nerf.train(args.epochs, data["train"])
    if args.save != "":
      nerf.save(args.save)
  else:
    nerf.load(args.model)

  height, width = 800, 800
  # render_gif(nerf, (height, width), 1111, 3, 4)
  # render_voxels(nerf, (1, 1, 1))
  voxel_map = generate_voxel_map(nerf, (1, 1, 1))
  render_voxels(voxel_map, "./results/voxel.png")
  generate_mc_schematic(voxel_map, './results', 'my-schem')  

