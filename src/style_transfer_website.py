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
image_quality_selection = st.sidebar.selectbox('Image Quality',
                                                ["Low", "Medium", "High"])
style_weight = st.sidebar.slider('Style Weight', 2, 8, step=1, value=4)
image_quality_mapping = {"Low": 128, "Medium": 256, "High": 512}
imsize = image_quality_mapping[image_quality_selection]
# imsize = 512 if torch.cuda.is_available() else 128
# imsize = 512
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
    style_image = Image.open(f'../imgs/{default_style_image_name}')
    content_image = Image.open(f'../imgs/{default_content_image_name}')
else:
    style_image = Image.open(style_image_upload)
    content_image = Image.open(content_image_upload)

# style_image_pil = Image.open(f'../imgs/{style_image_name}')
# content_image_pil = Image.open(content_image_name)
# st.image(style_image_pil)
# st.image(content_image_pil)

st.title('Content Image')
st.image(content_image.resize((444,444)))

st.title('Style Image')
st.image(style_image.resize((444,444)))

# style_image_loader = image_loader(f'../imgs/{style_image_name}', imsize, device)
style_image_loader = image_loader(style_image, imsize, device)
content_image_loader = image_loader(content_image, imsize, device)

# content_img = image_loader(content_image_name, imsize, device)

# assert style_img.size() == content_img.size()

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

st.write("Running Style Transfer")
# input_img = content_img.clone()
input_img = torch.randn(content_image_loader.data.size(), device=device)

# output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, input_img, style_weight=1e4)

go = st.button('Optimize this image')

out = st.empty()
iter_info = st.empty()

im = imshow(input_img)
out.image(im.resize((444,444)))


if go:
    for i, n in run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_image_loader, style_image_loader, input_img, style_weight=10 ** style_weight):
        im = imshow(i)
        out.image(im.resize((444,444)))
        iter_info.write(f'Iteration Number: {n}')
    # out.image(imshow(i.resize((444,444))))

# out = st.empty()
# out.image(imshow(output).resize((444,444)))
# st.image(imshow(output).resize((444,444)))