from simple_nerf import SimpleNERFModel
from data import get_dataloaders
from utils import *
if __name__ == '__main__':
  train, val, test = get_dataloaders('./data/hotdog')
  nerf = SimpleNERFModel()
  nerf.train(100, train, val)
  