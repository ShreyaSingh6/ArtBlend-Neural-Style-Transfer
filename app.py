import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
import io

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Image transformation
def load_image(image_file, max_size=400, shape=None):
    image = Image.open(image_file).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# Convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + \
            np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    return image

# Feature extraction
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Streamlit UI
st.title("🎨 Neural Style Transfer with PyTorch + Streamlit")

content_image = st.file_uploader("Upload a Content Image", type=["jpg", "png", "jpeg"])
style_image = st.file_uploader("Upload a Style Image", type=["jpg", "png", "jpeg"])

if content_image and style_image:
    content = load_image(content_image).to(device)
    style = load_image(style_image, shape=content.shape[-2:]).to(device)

    st.image([im_convert(content), im_convert(style)], caption=["Content", "Style"], width=300)

    if st.button("Generate Stylized Image"):
        with st.spinner("Stylizing... Please wait"):
            content_features = get_features(content, vgg)
            style_features = get_features(style, vgg)
            style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

            target = content.clone().requires_grad_(True).to(device)

            style_weights = {'conv1_1': 1., 'conv2_1': 0.75,
                             'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
            content_weight = 1
            style_weight = 1e9
            optimizer = optim.Adam([target], lr=0.003)
            steps = 300

            for ii in range(steps):
                target_features = get_features(target, vgg)
                content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
                style_loss = 0
                for layer in style_weights:
                    target_feature = target_features[layer]
                    target_gram = gram_matrix(target_feature)
                    style_gram = style_grams[layer]
                    _, d, h, w = target_feature.shape
                    layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                    style_loss += layer_style_loss / (d * h * w)
                total_loss = content_weight * content_loss + style_weight * style_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            st.success("Done!")
            st.image(im_convert(target), caption="Stylized Image", use_column_width=True)
