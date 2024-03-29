import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models
from style_transfer import (image_loader,
                            run_style_transfer,
                            imshow)
from PIL import Image
from io import BytesIO
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f'Running on: {device}')
image_quality_selection = st.sidebar.selectbox('Image Quality',
                                                ["Low", "Medium", "High", "Ultra High"])

preset_style_image_names = os.listdir("imgs/style_images")

#TODO: create style weight and iterations dictionary based on each style image for general best results

style_image_selection = st.sidebar.selectbox("Style Image",
                                                preset_style_image_names, index=3) #first default should probably be starry night
style_weight = st.sidebar.slider('Style Weight', 2, 7, step=1, value=5)
number_of_iterations = st.sidebar.slider('Number of Iterations', 150, 1000, 300, 50)

image_quality_mapping = {"Low": 128, "Medium": 256, "High": 512, "Ultra High": 1024}
imsize = image_quality_mapping[image_quality_selection]
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
default_style_image_name = preset_style_image_names[0]
default_content_image_name = 'bird.jpg'

#style image
st.title('Style Image')
style_image_upload = st.file_uploader('Style Image')
if style_image_upload:
    style_image = Image.open(style_image_upload)
else:
    style_image = Image.open(f"imgs/style_images/{style_image_selection}")
style_image_size = style_image.size
style_image_ratio = style_image_size[0] / style_image_size[1]
style_image_size_display = int(style_image_ratio * 444)
st.image(style_image.resize((style_image_size_display,444)))

st.title('Content Image')
content_image_upload = st.file_uploader('Content Image')
if content_image_upload:
    content_image = Image.open(content_image_upload)
else:
    content_image = Image.open(f"imgs/{default_content_image_name}")

content_image_size = content_image.size
content_image_ratio = content_image_size[0] / content_image_size[1]
content_image_size_display = int(content_image_ratio * 444)
st.image(content_image.resize((content_image_size_display,444)))

style_image_loader = image_loader(style_image, imsize, device)
content_image_loader = image_loader(content_image, imsize, device)

cnn = torch.load('models/VGG19.pt').to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

st.write("Running Style Transfer")
input_img = torch.randn(content_image_loader.data.size(), device=device)

go = st.button('Optimize this image')

stylized_image_placeholder = st.empty()
iter_placeholder = st.empty()

im = imshow(input_img)
stylized_image_placeholder.image(im.resize((content_image_size_display,444)))

if go:
    for i, n in run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_image_loader, style_image_loader, input_img,
                                num_steps=number_of_iterations-20, style_weight=10 ** style_weight):
        im = imshow(i)
        stylized_image_placeholder.image(im.resize((content_image_size_display,444)))
        iter_placeholder.write(f'Iteration Number: {n}')