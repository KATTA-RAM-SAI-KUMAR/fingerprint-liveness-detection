import os
import time

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.model import build_model

from flask import Flask, render_template, request, redirect, url_for, session

# ==================================================
# Flask App
# ==================================================

app = Flask(__name__)
app.secret_key = "fingersecure123"
UPLOAD_FOLDER = "static/uploads"

os.makedirs("static/uploads", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==================================================
# Configuration
# ==================================================

MODEL_PATH = "models/liveness_model.pth"

IMAGE_SIZE = 224

THRESHOLD = 0.02

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# Load Model
# ==================================================

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

# ==================================================
# Transform
# ==================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
# ==================================================
# Routes
# ==================================================

@app.route("/")
def home():

    return render_template("home.html")


@app.route("/detect")
def detect():

    return render_template(

        "detect.html",

        prediction=session.pop("prediction", None),

        status=session.pop("status", None),

        confidence=session.pop("confidence", None),

        live_score=session.pop("live_score", None),

        spoof_score=session.pop("spoof_score", None),

        inference=session.pop("inference", None),

        analysis=session.pop("analysis", None),

        image_path=session.pop("image_path", None)

    )


@app.route("/contact")
def contact():

    return render_template("contact.html")


# ==================================================
# Prediction
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template(
            "detect.html",
            error="No image uploaded."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "detect.html",
            error="Please select an image."
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = Image.open(filepath).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    start = time.perf_counter()

    with torch.no_grad():

        output = model(tensor)

        probability = F.softmax(output, dim=1)

    end = time.perf_counter()

    live_score = probability[0][0].item()

    spoof_score = probability[0][1].item()

    inference_time = (end - start) * 1000

    confidence = max(
        live_score,
        spoof_score
    ) * 100
    # ==================================================
    # Prediction Result
    # ==================================================

    if live_score > spoof_score:

        prediction = "LIVE"

        status = "🟢 Genuine Fingerprint"

        analysis = [
            "Natural ridge flow detected",
            "Texture consistency is normal",
            "No spoof artifacts found",
            "Authentication Approved"
        ]

    else:

        prediction = "SPOOF"

        status = "🔴 Spoof Fingerprint"

        analysis = [
            "Presentation attack detected",
            "Artificial ridge texture observed",
            "Possible print/display attack",
            "Authentication Rejected"
        ]
    session["prediction"] = prediction

    session["status"] = status

    session["confidence"] = round(confidence,2)

    session["live_score"] = round(live_score,4)

    session["spoof_score"] = round(spoof_score,4)

    session["inference"] = round(inference_time,2)

    session["analysis"] = analysis

    session["image_path"] = f"uploads/{file.filename}"

    return redirect(url_for("detect"))

# ==================================================
# Run Application
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )