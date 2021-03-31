import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models
from style_transfer import (image_loader,
                            run_style_transfer,
                            imshow)
from PIL import Image
from io import BytesIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f'Running on: {device}')
image_quality_selection = st.sidebar.selectbox('Image Quality',
                                                ["Low", "Medium", "High", "Ultra High"])
style_weight = st.sidebar.slider('Style Weight', 3, 7, step=1, value=4)
number_of_iterations = st.sidebar.slider('Number of Iterations', 150, 1000, 300, 50)

image_quality_mapping = {"Low": 128, "Medium": 256, "High": 512, "Ultra High": 1024}
imsize = image_quality_mapping[image_quality_selection]
# imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
default_style_image_name = 'Alex_Grey_Over_Soul.jpg'
default_content_image_name = 'bird.jpg'

style_image_upload = st.file_uploader('Style Image')
content_image_upload = st.file_uploader('Content Image')
if not style_image_upload and not content_image_upload:
    #if there are no images uploaded
    style_image = Image.open(f'imgs/{default_style_image_name}')
    content_image = Image.open(f'imgs/{default_content_image_name}')
else:
    style_image = Image.open(style_image_upload)
    content_image = Image.open(content_image_upload)

content_image_size = content_image.size
content_image_ratio = content_image_size[0] / content_image_size[1]
content_image_size_display = int(content_image_ratio * 444)

style_image_size = style_image.size
style_image_ratio = style_image_size[0] / style_image_size[1]
style_image_size_display = int(style_image_ratio * 444)

st.title('Content Image')
st.image(content_image.resize((content_image_size_display,444)))

st.title('Style Image')
st.image(style_image.resize((style_image_size_display,444)))

style_image_loader = image_loader(style_image, imsize, device)
content_image_loader = image_loader(content_image, imsize, device)

cnn = torch.load('models/VGG19.pt').to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

st.write("Running Style Transfer")
input_img = torch.randn(content_image_loader.data.size(), device=device)

go = st.button('Optimize this image')

out = st.empty()
iter_info = st.empty()

im = imshow(input_img)
out.image(im.resize((content_image_size_display,444)))

if go:
    for i, n in run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_image_loader, style_image_loader, input_img,
                                num_steps=number_of_iterations-20, style_weight=10 ** style_weight):
        im = imshow(i)
        out.image(im.resize((content_image_size_display,444)))
        iter_info.write(f'Iteration Number: {n}')
    # out.image(imshow(i.resize((444,444))))

# out = st.empty()
# out.image(imshow(output).resize((444,444)))
# st.image(imshow(output).resize((444,444)))