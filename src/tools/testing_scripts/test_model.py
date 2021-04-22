import torch

model_path = "/home/students/acct1001_05/CenterTrack_MOT_Paper/models/dla34up-cityscapes.pth"
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
# print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
# print(checkpoint)
state_dict_ = checkpoint['state_dict']
print(state_dict_)
# state_dict_ = checkpoint['state_dict']