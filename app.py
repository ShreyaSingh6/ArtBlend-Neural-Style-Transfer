import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch.optim as optim
import io

st.set_page_config(page_title="Neural Style Transfer", layout="centered")

# Title
st.title("🎨 Neural Style Transfer with VGG19")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader
def load_image(image, max_size=400, shape=None):
    image = Image.open(image).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Show image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + \
            np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    return image

# Get features
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # content layer
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Load VGG19
@st.cache_resource
def load_model():
    weights = VGG19_Weights.DEFAULT
    model = vgg19(weights=weights).features
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)

vgg = load_model()

# File Uploads
content_image = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'])
style_image = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'])

if content_image and style_image:
    content = load_image(content_image)
    style = load_image(style_image, shape=content.shape[-2:])

    st.subheader("🖼️ Content Image")
    st.image(im_convert(content), use_column_width=True)

    st.subheader("🎨 Style Image")
    st.image(im_convert(style), use_column_width=True)

    if st.button("Generate Stylized Image"):
        with st.spinner("Generating... Please wait ⏳"):
            target = content.clone().requires_grad_(True).to(device)

            style_weights = {'conv1_1': 1.0,
                             'conv2_1': 0.8,
                             'conv3_1': 0.5,
                             'conv4_1': 0.3,
                             'conv5_1': 0.1}

            content_weight = 1e4
            style_weight = 1e2

            content_features = get_features(content, vgg)
            style_features = get_features(style, vgg)

            style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

            optimizer = optim.Adam([target], lr=0.003)
            steps = 300

            for step in range(1, steps + 1):
                target_features = get_features(target, vgg)

                content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

                style_loss = 0
                for layer in style_weights:
                    target_feature = target_features[layer]
                    target_gram = gram_matrix(target_feature)
                    style_gram = style_grams[layer]
                    _, d, h, w = target_feature.shape
                    style_loss += style_weights[layer] * torch.mean((target_gram - style_gram) ** 2) / (d * h * w)

                total_loss = content_weight * content_loss + style_weight * style_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if step % 50 == 0:
                    st.info(f"Step [{step}/{steps}] - Loss: {total_loss.item():.2f}")

            st.success("Style transfer completed!")
            final_image = im_convert(target)
            st.image(final_image, caption="🖼️ Stylized Image", use_column_width=True)

            # Optional download
            result = Image.fromarray((final_image * 255).astype(np.uint8))
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button("📥 Download Stylized Image", data=byte_im, file_name="stylized.png", mime="image/png")
