import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    style_img = image_loader('../imgs/picasso.jpg')
    content_img = image_loader('../imgs/dancing.jpg')

    assert style_img.size() == content_img.size()

    unloader = transforms.ToPILImage()
    plt.ion()

    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')