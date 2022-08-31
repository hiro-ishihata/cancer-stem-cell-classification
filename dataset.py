# -*- coding: utf-8 -*- 
import sys, os, time, glob
import argparse
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

def showim(ds, nx=8, ny=4, msg=None, fname=None): # 画像データの表示  
    """ make nx*ny grid images from list of torch image
    ds: list of image tensor (c,h,w),  0-1 float
    nx, ny : image grid size
    msg : list of str for each image
    fname : save filename if not None
    """
    b = len(ds)
    c,h,w = ds[0].shape
    nn = nx*ny
    canvas = Image.new("RGB", (w*nx, h*ny), (0,0,0))
    for i in range(nn):
        x,y = i % nx, i // nx    # display position
        if i >= len(ds):
            break
        npimg = np.asarray(ds[i])*255
        npimg = np.clip(npimg, 0, 255).astype(np.uint8)
        npimg = np.transpose(npimg, (1,2,0))   # convert to (b,h,w,c)
        pilimg = Image.fromarray(npimg) # convert to PIL
        if msg is not None:
            draw = ImageDraw.Draw(pilimg)              
            draw.text((10,10), msg[i], fill='#fff')
        canvas.paste(pilimg,(x*w, y*h))
    if fname: canvas.save(fname) # save to file
    else: canvas.show()          # display


def get_transform(mode='train'):
    """ image transform pipline for data augumentation
    Args:
        mode : 'train' for training, 'test' for validation and test, other for display
    Returns:
        transform function
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomRotation(30,resample=Image.BILINEAR),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
    elif mode == 'test': # test mode
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
    else: # non normalize
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


class CSCDataset(Dataset):
    """ Cancer Stem Cell Picture dataset
    """
    def __init__(self, transfunc=None, folds=None, seed=None):
        """
        Args:
            transfunc : data transform function
            folds : list of using group id, data is divided int 5 group (0 to 4)
            seed : random number generator seed
        """
        base = './sampledata/'  # image file directory
        print(f"load dataset start folds {folds}")
        self.transfunc = transfunc  # data augmentation function
        self.image = [] # hold image data
        self.label = [] # hold label
        ngroup = 5
        rng = np.random.default_rng(seed)        
        img0 = sorted(glob.glob(base+'/day1/*.png')) # class 0
        img1 = sorted(glob.glob(base+'/day2/*.png')) # class 1
        for kind, imgs in enumerate([img0, img1]): # do for all dirextories
            fold = rng.integers(ngroup, size=len(imgs)) # ngroup fold value
            for i, imf in enumerate(imgs):  # process each image file
                if folds is not None and fold[i] not in folds:   # select fold
                    continue
                try:
                    tmp = Image.open(imf) # read image
                except: # skip if error 
                     continue
                if tmp.mode != 'RGB': # convert to RGB if not RGB
                    tmp = tmp.convert("RGB")
                img = tmp.copy()
                tmp.close() # avoid Too many open file
                self.image.append(img)   # add PIL image
                self.label.append(kind)  # add label
        print(f"dataset len={len(self.image)}")
        
    def __len__(self):
        return len(self.image)
    def __getitem__(self, i):
        """ return i-th data (image and label)
        """
        if self.transfunc is not None: # data augmentation
            x = self.transfunc(self.image[i])
        else:
            x = self.image[i]
        t = self.label[i]
        return x, t

if __name__ == "__main__":
    """ simple test dor CSCDataset 
    """
    d = CSCDataset(transfunc=get_transform('train'),
                   seed=123, folds=[0,1,2,3])
    td = CSCDataset(transfunc=get_transform('test'),
                    seed=123, folds=[4])
    
    dl1 = DataLoader(d, batch_size=64, shuffle=True)
    for x, t in dl1:
        gt = [f"GT={tt.item()}" for tt in t]
        showim(x, msg=gt) # show first batch
        break
    dl2 = DataLoader(td, batch_size=64, shuffle=True)
    for x, t in dl2:
        gt = [f"GT={tt.item()}" for tt in t]
        showim(x, msg=gt)
        break
        

