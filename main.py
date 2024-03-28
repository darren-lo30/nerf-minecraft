from simple_nerf import SimpleNERFModel
from data import load_data
from utils import get_device

if __name__ == "__main__":
  data = load_data("./data/hotdog")
  nerf = SimpleNERFModel(device=get_device())
  # 100 train images
  # ~ 1000 epochs (over 100 images)
  num_epochs = 100000
  nerf.train(num_epochs, data["train"], data["val"])
