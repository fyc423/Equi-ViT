import torch
from typing import Iterable
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import torchio as tio
from PIL import Image


class DummyToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)

class AddBottomLine(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        _, H, W = x.shape
        white_value = 255 if x.dtype == torch.uint8 else 1

        x[:, int(0.9*H), int(0.2*W):int(0.8*W)] = white_value
        return x
    
    
class PadTransWrapper(object):
    def __init__(self, trans, padding="constant", img_size=32):
        super().__init__()
        self.trans = trans
        self.padding_mode = padding
        if padding != "constant":
            self.to_pad = int(img_size // 2)
        else:
            self.to_pad = None
        self.img_size = img_size
        
    def __call__(self, x):
        # pad the image before rotate
        if self.to_pad != None:
            x = F.pad(x, (self.to_pad, self.to_pad, self.to_pad, self.to_pad), padding_mode=self.padding_mode)
        x = self.trans(x)
        # crop the image after rotate
        if self.to_pad != None:
            x = F.center_crop(x, [self.img_size, self.img_size])
        return x
    


class FixRotate(object):
    def __init__(self, degree, expand=False, interpolation="bilinear"):
        super().__init__()
        self.degree = degree
        self.expand = expand
        if interpolation == "nearest":
            self.interpolation = F.InterpolationMode.NEAREST
        elif interpolation == "bilinear":
            self.interpolation = F.InterpolationMode.BILINEAR
        elif interpolation == "bicubic":
            self.interpolation = F.InterpolationMode.BICUBIC

    def __call__(self, x):
        return F.rotate(x, self.degree, expand=self.expand, 
                        interpolation=self.interpolation)



def get_nct_crc_transforms(args):

    transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
        
    test_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=224))

    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform



