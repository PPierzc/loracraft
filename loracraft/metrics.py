import torch
from torch import nn
from torchvision.models import inception_v3, resnet18
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
import numpy as np
from transformers import CLIPProcessor, CLIPModel

def calculate_activation_statistics(images, model, batch_size=64, dims=512):
    model.eval()
    n_batches = len(images) // batch_size
    if not n_batches:
        n_batches = 1
    act = np.zeros((len(images), dims))

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end].cuda()

        with torch.no_grad():
            pred = model(batch)

        act[start:end] = pred.cpu().numpy()

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    sqrt_term, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
    if not np.isfinite(sqrt_term).all():
        offset = np.eye(sigma1.shape[0]) * eps
        sqrt_term = sqrtm(np.dot(sigma1 + offset, sigma2 + offset))

    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * sqrt_term)
    return fid.real

def preprocess_images(images):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    return torch.stack([preprocess(img) for img in images])

def fid_score(images1, images2, batch_size=16):
    # Load Inception-v3 model pretrained on ImageNet
    inception_model = resnet18(pretrained=True).cuda()
    inception_model.fc = torch.nn.Identity()

    # Preprocess images
    images1 = preprocess_images(images1)
    images2 = preprocess_images(images2)

    # Calculate Inception-v3 features and statistics
    mu1, sigma1 = calculate_activation_statistics(images1, inception_model, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, inception_model, batch_size)

    # Calculate FID score
    fid = calculate_fid(mu1, sigma1, mu2, sigma2)

    return fid

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_score(true_images, gen_images, n_classes=4):
    inputs = processor(text=[""], images=true_images + gen_images, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    outputs.image_embeds.shape

    scores = torch.cosine_similarity(outputs.image_embeds[:len(true_images)].unsqueeze(1), outputs.image_embeds[len(true_images):], dim=-1)
    #scores = scores.reshape(len(true_images), n_classes, -1).mean([0, 2])
    return float(scores[0][0].detach())