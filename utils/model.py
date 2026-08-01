import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def build_model():

    model = mobilenet_v3_small(weights="DEFAULT")

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        in_features,
        2
    )

    return model