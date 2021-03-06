import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import time

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                                content_layers=['conv_4'], 
                                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img,
                        input_img, num_steps=300, style_weight=1e6, content_weight=1):
    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                    style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run}:")
                print(f"Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}")
                print()

            return style_score + content_score
        optimizer.step(closure)
        input_img.data.clamp_(0, 1)
        yield input_img

    # input_img.data.clamp_(0, 1)

    # return input_img


def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(image_name)
    image = image.resize((444, 444))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
    # plt.imshow(image)
    # if title:
    #     plt.title(title)
    # plt.pause(0.001)

if __name__ == "__main__":
    start = time.time()
    # plt.ion()
    plt.ioff()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imsize = 512 if torch.cuda.is_available() else 128
    # imsize = 512
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    style_image_name = 'Alex_Grey_Over_Soul.jpg'
    # content_image_name = 'dancing.jpg'
    content_image_name = '../../../../downloads/1200px-Eopsaltria_australis_-_Mogo_Campground.jpg'
    style_img = image_loader(f'../imgs/{style_image_name}')
    content_img = image_loader(content_image_name)

    assert style_img.size() == content_img.size()

    

    # plt.figure()
    # imshow(style_img, title='Style Image')

    # plt.figure()
    # imshow(content_img, title='Content Image')

    cnn = models.vgg19(pretrained=True).features.to(device).eval() #pretrained on imagenet

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone()
    # input_img = torch.randn(content_img.data.size(), device=device)

    # plt.figure()
    # imshow(input_img, title='Input Image')

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, style_weight=1e4)
    end = time.time()
    print(f"Time taken to train: {(end-start)/60} minutes")
    plt.figure()
    imshow(output, title='Output Image')
    plt.savefig(f'Styled input image bird + {style_image_name} less style.png')
    plt.show()