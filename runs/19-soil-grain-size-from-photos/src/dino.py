"""DINOv2 feature extraction (CPU)."""
import torch
import numpy as np
import timm
import torchvision.transforms as T
from PIL import Image

_model = None
_tf = None


def get_dino(size=518):
    global _model, _tf
    if _model is None:
        torch.set_num_threads(8)
        _model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        _model.eval()
    if _tf is None:
        _tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return _model, _tf


@torch.no_grad()
def dino_embed(path, model=None, tf=None, size=518):
    if model is None or tf is None:
        model, tf = get_dino(size)
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    feats = model.forward_features(x).squeeze(0)  # (n_tokens, D), CLS first
    cls = feats[0].numpy().astype(float)
    patch = feats[1:].mean(0).numpy().astype(float)
    return np.concatenate([cls, patch]).astype(float)
