import base64
import io
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageOps
from transformers import ViTImageProcessor, ViTModel

# ---------------------------------------------------------
# 1. Model & Engine Definitions
# ---------------------------------------------------------

class KeratoconusMultimodalNet(nn.Module):
    """Late-fusion multimodal model using a ViT image stream and a tabular MLP stream."""

    def __init__(self, tabular_input_dim: int, num_classes: int = 2):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        image_feature_dim = self.vit.config.hidden_size
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        combined_dim = image_feature_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        vit_outputs = self.vit(pixel_values=images)
        image_features = vit_outputs.last_hidden_state[:, 0, :]
        tabular_features = self.tabular_mlp(tabular_data)
        fused_features = torch.cat((image_features, tabular_features), dim=1)
        return self.classifier(fused_features)


class EyeNovaApp:
    """Simple wrapper around static assets and model inference helpers."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.image_processor = None
        self._loaded = False

    def load_artifacts(self) -> None:
        if self._loaded:
            return
        
        # Paths relative to the app directory
        WORKDIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(WORKDIR, 'keratoconus_multimodal_model.pt')
        SCALER_PATH = os.path.join(WORKDIR, 'scaler (1).pkl')
        LABEL_ENCODER_PATH = os.path.join(WORKDIR, 'label_encoder.pkl')
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = KeratoconusMultimodalNet(tabular_input_dim=6, num_classes=2).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        with open(SCALER_PATH, 'rb') as handle:
            self.scaler = pickle.load(handle)

        with open(LABEL_ENCODER_PATH, 'rb') as handle:
            self.label_encoder = pickle.load(handle)

        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self._loaded = True

    def preprocess_tabular(self, payload: dict) -> torch.Tensor:
        gender_value = payload.get('gender', '0')
        try:
            gender_encoded = int(gender_value)
        except ValueError:
            gender_encoded = 1 if str(gender_value).lower() in {'male', 'm', '1'} else 0

        row = {
            'age_years': float(payload.get('age', 0) or 0),
            'gender': gender_encoded,
            'astig_value_D': float(payload.get('astigmatism_value', 0) or 0),
            'astig_axis_deg': float(payload.get('astigmatism_axis', 0) or 0),
            'pachy_thinnest_x': float(payload.get('pachy_x', 0) or 0),
            'pachy_thinnest_y': float(payload.get('pachy_y', 0) or 0),
        }
        frame = np.array([[row[col] for col in [
            'age_years',
            'gender',
            'astig_value_D',
            'astig_axis_deg',
            'pachy_thinnest_x',
            'pachy_thinnest_y',
        ]]], dtype=np.float32)
        scaled = self.scaler.transform(frame)
        return torch.tensor(scaled, dtype=torch.float32)

    def generate_isolated_heatmap(self, stitched_image: Image.Image, saliency_map: np.ndarray, threshold: float = 0.35) -> Image.Image:
        width, height = stitched_image.size
        right_panel = np.full((height, width, 3), 255, dtype=np.uint8)
        attention_mask = saliency_map >= threshold
        
        v = saliency_map
        r = np.zeros_like(v)
        g = np.zeros_like(v)
        b = np.zeros_like(v)
        
        m1 = v < 0.25
        b[m1] = 0.5 + 2.0 * v[m1]
        g[m1] = 4.0 * v[m1]
        
        m2 = (v >= 0.25) & (v < 0.5)
        g[m2] = 1.0
        b[m2] = 1.0 - 4.0 * (v[m2] - 0.25)
        
        m3 = (v >= 0.5) & (v < 0.75)
        r[m3] = 4.0 * (v[m3] - 0.5)
        g[m3] = 1.0
        
        m4 = v >= 0.75
        r[m4] = 1.0
        g[m4] = 1.0 - 4.0 * (v[m4] - 0.75)
        
        r_img = (np.clip(r, 0.0, 1.0) * 255.0).astype(np.uint8)
        g_img = (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)
        b_img = (np.clip(b, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        mapped_heatmap = np.stack([r_img, g_img, b_img], axis=-1)
        right_panel[attention_mask] = mapped_heatmap[attention_mask]
        
        return Image.fromarray(right_panel, mode='RGB')

# ---------------------------------------------------------
# 2. Gradio App Interface logic
# ---------------------------------------------------------

engine = EyeNovaApp()

def predict_gradio(age, gender, astigmatism_value, astigmatism_axis, pachy_x, pachy_y, ant_img, ax_img, pachy_img, post_img):
    if ant_img is None or ax_img is None or pachy_img is None or post_img is None:
        return "Error: All 4 topography maps are required.", "", None, None
    
    engine.load_artifacts()
    
    payload = {
        'age': age,
        'gender': gender,
        'astigmatism_value': astigmatism_value,
        'astigmatism_axis': astigmatism_axis,
        'pachy_x': pachy_x,
        'pachy_y': pachy_y,
    }
    
    # Process images
    target_size = (112, 112)
    images = []
    for img in [ant_img, ax_img, pachy_img, post_img]:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        pil_image = img.convert('RGB').resize(target_size)
        images.append(pil_image)
        
    canvas = Image.new('RGB', (224, 224), color=(255, 255, 255))
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (112, 0))
    canvas.paste(images[2], (0, 112))
    canvas.paste(images[3], (112, 112))
    
    pixel_values = engine.image_processor(images=canvas, return_tensors='pt')['pixel_values']
    tabular = engine.preprocess_tabular(payload)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.set_grad_enabled(True):
        pixel_values = pixel_values.to(device)
        tabular = tabular.to(device)
        pixel_values.requires_grad_(True)
        
        vit_outputs = engine.model.vit(pixel_values=pixel_values, return_dict=True)
        image_features = vit_outputs.last_hidden_state[:, 0, :]
        tabular_features = engine.model.tabular_mlp(tabular)
        fused_features = torch.cat((image_features, tabular_features), dim=1)
        logits = engine.model.classifier(fused_features)
        probabilities = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, pred_idx].item() * 100.0)
        
        if pred_idx == 1:
            status = 'Keratoconus Detected'
        else:
            status = 'Normal Cornea'
            
        loss = logits[:, pred_idx].sum()
        loss.backward(retain_graph=True)
        
        pixel_grads = torch.abs(pixel_values.grad)[0]
        if pixel_grads is not None and pixel_grads.numel() > 0:
            pixel_grads = pixel_grads.detach().cpu()
            saliency_map = pixel_grads.mean(dim=0).numpy()
            saliency_map = np.clip((saliency_map - saliency_map.min()) / max(saliency_map.max() - saliency_map.min(), 1e-6), 0.0, 1.0)
        else:
            # Fallback
            patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]
            patch_importance = patch_embeddings.abs().mean(dim=2)[0].detach().cpu().reshape(14, 14)
            saliency_map = torch.nn.functional.interpolate(
                patch_importance.unsqueeze(0).unsqueeze(0).float(),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ).squeeze().numpy()
            saliency_map = np.clip((saliency_map - saliency_map.min()) / max(saliency_map.max() - saliency_map.min(), 1e-6), 0.0, 1.0)
            
        right_panel_pil = engine.generate_isolated_heatmap(canvas, saliency_map, threshold=0.35)
        
    return status, f"{confidence:.1f}%", canvas, right_panel_pil

# ---------------------------------------------------------
# 3. Gradio Interface Layout
# ---------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 👁️ Live Demo: Clinical AI Decision Support")
    gr.Markdown("Clinical AI decision support for keratoconus triage, combining multimodal topography and structured ocular parameters.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Clinical Inputs")
            age = gr.Number(label="Age (years)", value=25)
            gender = gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Male")
            astigmatism_value = gr.Number(label="Astigmatism Value (D)", value=0.0)
            astigmatism_axis = gr.Number(label="Astigmatism Axis (°)", value=0)
            pachy_x = gr.Number(label="Pachymetry Thinnest X Coordinate", value=0.0)
            pachy_y = gr.Number(label="Pachymetry Thinnest Y Coordinate", value=0.0)
            
        with gr.Column():
            gr.Markdown("### Topography Maps (Upload all 4)")
            ant_img = gr.Image(label="Anterior Map", type="pil")
            ax_img = gr.Image(label="Axial Map", type="pil")
            pachy_img = gr.Image(label="Pachymetry Map", type="pil")
            post_img = gr.Image(label="Posterior Map", type="pil")
            
    submit_btn = gr.Button("Run Assessment", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Results")
            status_out = gr.Textbox(label="Diagnosis")
            confidence_out = gr.Textbox(label="Confidence")
        with gr.Column():
            gr.Markdown("### Visualizations")
            stitched_out = gr.Image(label="Stitched Topography Grid")
            heatmap_out = gr.Image(label="AI Attention Heatmap")
            
    submit_btn.click(
        fn=predict_gradio,
        inputs=[age, gender, astigmatism_value, astigmatism_axis, pachy_x, pachy_y, ant_img, ax_img, pachy_img, post_img],
        outputs=[status_out, confidence_out, stitched_out, heatmap_out]
    )

if __name__ == '__main__':
    demo.launch()
