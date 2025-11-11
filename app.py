import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import zipfile
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Optional: unzip examples if present ---
if os.path.exists("examples.zip"):
    with zipfile.ZipFile("examples.zip", "r") as zip_ref:
        zip_ref.extractall(".")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Face detector
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE)
mtcnn.eval()

# Base model (binary head)
model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1
)

# Load your trained checkpoint (if you trained the classifier head)
# Replace filename if needed.
checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE).eval()

# Build examples list (up to 10)
EXAMPLES_FOLDER = "examples"
examples = []
if os.path.isdir(EXAMPLES_FOLDER):
    for name in os.listdir(EXAMPLES_FOLDER):
        path = os.path.join(EXAMPLES_FOLDER, name)
        if os.path.isfile(path):
            label = name.split("_")[0]
            examples.append([path, label])
    np.random.shuffle(examples)
    examples = examples[:10]


def predict(input_image: Image.Image, true_label: str):
    # Face detect + crop
    face = mtcnn(input_image)
    if face is None:
        raise gr.Error("No face detected")
    face = face.unsqueeze(0)  # [1, C, H, W]
    face = F.interpolate(face, size=(256, 256), mode="bilinear", align_corners=False)

    # Keep a uint8 copy for overlay
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    prev_face = np.clip(prev_face, 0, 255).astype("uint8")

    # Normalize for model + CAM
    face = (face.to(torch.float32) / 255.0).to(DEVICE)
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    face_image_to_plot = np.clip(face_image_to_plot, 0.0, 1.0)

    # Grad-CAM on a late conv layer
    target_layers = [model.block8.branch1[-1]]
    targets = [ClassifierOutputTarget(0)]
    # NOTE: use context manager; don't pass use_cuda (removed in grad-cam 1.5.5)
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0]

    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    vis_uint8 = (np.clip(visualization, 0, 1) * 255).astype("uint8")
    face_with_mask = cv2.addWeighted(prev_face, 1.0, vis_uint8, 0.5, 0)

    # Prediction (your training: 1 => fake)
    with torch.no_grad():
        logit = model(face).squeeze(0)           # scalar
        prob_fake = torch.sigmoid(logit).item()
        prob_real = 1.0 - prob_fake
        prediction = "real" if prob_fake < 0.5 else "fake"
        confidences = {"real": prob_real, "fake": prob_fake}

    # Return: (Label dict), (echoed text), (image)
    return confidences, true_label, face_with_mask


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="Ground Truth Label"),
    ],
    outputs=[
        gr.Label(label="Class"),
        gr.Textbox(label="Echoed Label"),
        gr.Image(label="Face with Explainability", type="numpy"),
    ],
    examples=examples
)

if __name__ == "__main__":
    demo.launch()
