from simple_nerf import SimpleNERFModel
from data import load_data
from utils import get_device
from render import render_gif, render_img
import torch
from torchvision.utils import save_image
import os
if __name__ == "__main__":
  data = load_data("./data/lego", scale_ratio=0.1)
  nerf = SimpleNERFModel(device=get_device())
  # 100 train images
  # ~ 1000 epochs (over 100 images)
  # num_epochs = 1000
  # nerf.train(num_epochs, data["train"])
  # nerf.save('./models/nerf')
  nerf.load('./models/nerf')
  # transform = torch.tensor([[-0.5469, -0.5920,  0.5920,  2.5115],
  #                           [ 0.8372, -0.3868,  0.3868,  1.6408],
  #                           [-0.0000, -0.7071, -0.7071,  3.0000],
  #                           [ 0.0000,  0.0000,  0.0000,  1.0000]])
  # img_tensor = render_img(nerf, (80, 80), 110, transform)
  # save_image(img_tensor, os.path.join('./results', "img.png"))
  render_gif(nerf, (80, 80), 110, 3, 3)
  