import base64
import io
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, render_template_string, request
from PIL import Image, ImageOps
from transformers import ViTImageProcessor, ViTModel


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


WORKDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKDIR, 'keratoconus_multimodal_model.pt')
SCALER_PATH = os.path.join(WORKDIR, 'scaler (1).pkl')
LABEL_ENCODER_PATH = os.path.join(WORKDIR, 'label_encoder.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def _load_image_from_memory(self, file_storage) -> np.ndarray:
        image_bytes = file_storage.read()
        if not image_bytes:
            raise ValueError('Empty image upload received.')
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = ImageOps.exif_transpose(image)
        return np.array(image)

    def preprocess_images(self, image_files: List[object]) -> torch.Tensor:
        target_size = (112, 112)
        images = []
        for item in image_files:
            arr = self._load_image_from_memory(item)
            pil_image = Image.fromarray(arr).resize(target_size)
            images.append(pil_image)

        canvas = Image.new('RGB', (224, 224), color=(255, 255, 255))
        canvas.paste(images[0], (0, 0))
        canvas.paste(images[1], (112, 0))
        canvas.paste(images[2], (0, 112))
        canvas.paste(images[3], (112, 112))

        processed = self.image_processor(images=canvas, return_tensors='pt')
        return processed['pixel_values']

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

    def infer(self, payload: dict, image_files: List[object]) -> Tuple[str, float, str]:
        self.load_artifacts()
        pixel_values = self.preprocess_images(image_files)
        tabular = self.preprocess_tabular(payload)
        with torch.set_grad_enabled(True):
            pixel_values = pixel_values.to(DEVICE)
            tabular = tabular.to(DEVICE)
            pixel_values.requires_grad_(True)

            vit_outputs = self.model.vit(pixel_values=pixel_values, output_attentions=True, return_dict=True)
            image_features = vit_outputs.last_hidden_state[:, 0, :]
            tabular_features = self.model.tabular_mlp(tabular)
            fused_features = torch.cat((image_features, tabular_features), dim=1)
            logits = self.model.classifier(fused_features)
            probabilities = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, pred_idx].item() * 100.0)

            if pred_idx == 1:
                label = 'Keratoconus Detected'
            else:
                label = 'Normal Cornea'

            attention = vit_outputs.attentions[-1]
            cls_attention = attention[:, :, 0, 1:].mean(dim=1)
            cls_attention.retain_grad()
            loss = logits[:, pred_idx].sum()
            loss.backward(retain_graph=True)
            guided_grads = torch.abs(cls_attention.grad)
            patch_importance = guided_grads.mean(dim=1)[0].detach().cpu()
            patch_importance = patch_importance.reshape(14, 14)
            heatmap = F.interpolate(
                patch_importance.unsqueeze(0).unsqueeze(0).float(),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ).squeeze().numpy()
            heatmap = np.clip((heatmap - heatmap.min()) / max(heatmap.max() - heatmap.min(), 1e-6), 0.0, 1.0)
            heatmap = (heatmap * 255).astype(np.uint8)
            overlay = Image.fromarray(heatmap, mode='L').resize((224, 224))
            base = Image.new('RGB', (224, 224), color=(255, 255, 255))
            color_overlay = Image.merge('RGB', (overlay, Image.new('L', overlay.size, 0), Image.new('L', overlay.size, 0)))
            base = Image.blend(base, color_overlay, alpha=0.35)
            buffer = io.BytesIO()
            base.save(buffer, format='PNG')
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

        return label, confidence, heatmap_b64


engine = EyeNovaApp()


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/predict')
def predict():
    try:
        engine.load_artifacts()
        required_fields = ['age', 'gender', 'astigmatism_value', 'astigmatism_axis', 'pachy_x', 'pachy_y']
        payload = {}
        for field in required_fields:
            value = request.form.get(field, '').strip()
            if not value:
                raise ValueError(f'Missing required field: {field}')
            payload[field] = value

        image_files = [
            request.files.get('anterior_map'),
            request.files.get('axial_map'),
            request.files.get('pachymetry_map'),
            request.files.get('posterior_map'),
        ]
        if any(item is None or item.filename == '' for item in image_files):
            raise ValueError('All four topography maps must be uploaded.')

        payload = {
            'age': payload['age'],
            'gender': payload['gender'],
            'astigmatism_value': payload['astigmatism_value'],
            'astigmatism_axis': payload['astigmatism_axis'],
            'pachy_x': payload['pachy_x'],
            'pachy_y': payload['pachy_y'],
        }
        status, confidence, heatmap_b64 = engine.infer(payload, image_files)
        return render_template('result.html', status=status, confidence=f'{confidence:.1f}', heatmap_b64=heatmap_b64)
    except Exception as exc:
        return render_template_string("<h2>Assessment Error</h2><p>{{ error }}</p><a href='/'>Return</a>", error=str(exc))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
