import torch
import torch.utils.data
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
import os
from opts import opts


def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  print('Creating model...')
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

  optimizer = get_optimizer(opt, model)

  if opt.load_model != '':
      model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)


if __name__ == "__main__":
  opt = opts().parse()
  main(opt)