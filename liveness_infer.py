import argparse
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.model import build_model

# ----------------------------
# Configuration
# ----------------------------

MODEL_PATH = "models/liveness_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.02

# ----------------------------
# Transform
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Load Model
# ----------------------------

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

# ----------------------------
# Predict Function
# ----------------------------

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    start = time.perf_counter()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)

    end = time.perf_counter()

    live_probability = probabilities[0][0].item()

    spoof_probability = probabilities[0][1].item()

    inference_time = (end - start) * 1000

    if live_probability >= THRESHOLD:

        prediction = "LIVE ✅"

        confidence = live_probability

    else:

        prediction = "SPOOF ❌"

        confidence = spoof_probability

    print()

    print("=" * 60)

    print("Fingerprint Liveness Detection")

    print("=" * 60)

    print(f"Prediction      : {prediction}")

    print(f"Confidence      : {confidence*100:.2f}%")

    print(f"LIVE Score      : {live_probability:.4f}")

    print(f"SPOOF Score     : {spoof_probability:.4f}")

    print(f"Threshold       : {THRESHOLD:.2f}")

    print(f"Inference Time  : {inference_time:.2f} ms")

    print("=" * 60)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Image Path"
    )

    args = parser.parse_args()

    predict(args.image)