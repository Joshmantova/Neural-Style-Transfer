import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models
from style_transfer import (image_loader,
                            run_style_transfer,
                            imshow)
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
style_image_name = 'Alex_Grey_Over_Soul.jpg'
content_image_name = '../../../../downloads/bird.jpg'
style_image_pil = Image.open(f'../imgs/{style_image_name}')
content_image_pil = Image.open(content_image_name)
st.image(style_image_pil)
st.image(content_image_pil)

style_img = image_loader(f'../imgs/{style_image_name}', imsize, device)
content_img = image_loader(content_image_name, imsize, device)

assert style_img.size() == content_img.size()

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

st.write("Running Style Transfer")
input_img = content_img.clone()
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, style_weight=1e4)
# out = st.empty()
# for i in run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, input_img, style_weight=1e4):
#     out.image(imshow(i.resize((444,444))))

# out = st.empty()
# out.image(imshow(output).resize((444,444)))
st.image(imshow(output).resize((444,444)))