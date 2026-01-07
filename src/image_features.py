import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from tqdm import tqdm


def load_cnn_model(device):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.eval()
    model.to(device)
    return model


def extract_image_features(image_dir, ids, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = load_cnn_model(device)

    features = []

    for img_id in tqdm(ids):
        img_path = os.path.join(image_dir, f"{img_id}.png")

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(image)
            feat = feat.view(feat.size(0), -1)

        features.append(feat.cpu().numpy().flatten())

    return features
