import argparse
from torchvision import datasets
from torchvision import transforms
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/home/yuchen/School/ift6135/celeba_hw4')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_dset = datasets.ImageFolder(root=os.path.join(args.data, 'train'), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)

test_dset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)
test_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=False)

for img, _ in train_loader:
    #TODO: training here
    pass

for img, _ in test_loader:
    #TODO: test here
    pass

