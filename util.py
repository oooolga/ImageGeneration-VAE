import torch
from vae import *

def load_data(args):

    from torchvision import datasets
    from torchvision import transforms
    import os, copy, pdb

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),
    								  transform=transform)
    anchor_pt = int(len(train_dset) * (1-args.valid_prop))
    valid_dset = copy.deepcopy(train_dset)

    train_dset.imgs = train_dset.imgs[:anchor_pt]
    valid_dset.imgs = valid_dset.imgs[anchor_pt:]

    train_loader = torch.utils.data.DataLoader(train_dset,
    										   batch_size=args.batch_size,
    										   shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_dset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'),
    								 transform=transform)
    test_loader = torch.utils.data.DataLoader(train_dset,
    										  batch_size=args.batch_size,
    										  shuffle=False)

    print('Finished loading data...')
    return train_loader, valid_loader, test_loader

def get_model(mode):
    if mode == 'deconvolution':
        model = VariationalAutoEncoder()
    else:
        model = VariationalUpsampleEncoder(mode='bilinear')

    if torch.cuda.is_available():
        model = model.cuda()
    return model
 