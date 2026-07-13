import base64
import io
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template_string, request
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


HTML_FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EyeNova • Clinical AI Decision Support</title>
  <style>
    :root { --bg:#f4f7fb; --panel:#ffffff; --ink:#10324a; --muted:#617688; --accent:#1f5d8a; --accent2:#0b6ea8; --line:#dbe4ee; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Segoe UI,Roboto,Arial,sans-serif; background:linear-gradient(135deg,#eef4fb,#f7fbff); color:var(--ink); }
    .shell { max-width:1180px; margin:0 auto; padding:28px; }
    .hero { background:linear-gradient(120deg,#0f2f44,#1a4e70); color:white; border-radius:24px; padding:28px 32px; box-shadow:0 20px 45px rgba(15,47,68,.16); }
    .hero h1 { margin:0 0 8px; font-size:2rem; }
    .hero p { margin:0; color:#dce9f3; max-width:700px; line-height:1.6; }
    .grid { display:grid; grid-template-columns:1.2fr .8fr; gap:24px; margin-top:24px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:24px; box-shadow:0 12px 30px rgba(16,50,74,0.06); }
    label { display:block; font-weight:600; margin-bottom:8px; color:var(--ink); }
    input, select, button { width:100%; border-radius:12px; border:1px solid var(--line); padding:12px 14px; font-size:15px; }
    input:focus, select:focus { outline:2px solid rgba(31,93,138,.2); border-color:var(--accent); }
    .row { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-bottom:14px; }
    .field { margin-bottom:14px; }
    .upload-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .upload-card { border:1px dashed var(--line); border-radius:14px; padding:12px; background:#fbfdff; }
    .upload-card small { display:block; color:var(--muted); margin-top:6px; }
    .preview { margin-top:10px; min-height:90px; display:flex; align-items:center; justify-content:center; border-radius:10px; border:1px solid var(--line); background:white; overflow:hidden; }
    .preview img { max-width:100%; max-height:140px; object-fit:cover; display:block; }
    .preview .placeholder { color:var(--muted); font-size:13px; text-align:center; padding:10px; }
    .btn { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:white; border:none; cursor:pointer; font-weight:700; transition:transform .15s ease; }
    .btn:hover { transform:translateY(-1px); }
    .btn.secondary { background:white; color:var(--accent); border:1px solid var(--line); }
    .result-shell { display:grid; gap:20px; }
    .metric { display:flex; justify-content:space-between; padding:12px 14px; background:#f7fbff; border-radius:12px; border:1px solid var(--line); }
    .heatmap { border-radius:18px; border:1px solid var(--line); background:#f9fcff; padding:14px; }
    .heatmap img { width:100%; border-radius:12px; display:block; }
    .status-badge { display:inline-block; padding:8px 12px; border-radius:999px; font-weight:700; background:#e7f4ff; color:var(--accent); }
    @media (max-width:900px){ .grid,.upload-grid,.row{grid-template-columns:1fr;} }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>EyeNova</h1>
      <p>Clinical AI decision support for keratoconus triage, combining multimodal topography and structured ocular parameters into a single diagnostic view.</p>
    </div>
    <div class="grid">
      <div class="card">
        <h2 style="margin-top:0">Clinical Input Form</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <div class="row">
            <div class="field">
              <label for="age">Age (years)</label>
              <input id="age" name="age" type="number" min="1" max="120" step="1" required />
            </div>
            <div class="field">
              <label for="gender">Gender</label>
              <select id="gender" name="gender" required>
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="astigmatism_value">Astigmatism Value (D)</label>
              <input id="astigmatism_value" name="astigmatism_value" type="number" step="0.01" required />
            </div>
            <div class="field">
              <label for="astigmatism_axis">Astigmatism Axis (°)</label>
              <input id="astigmatism_axis" name="astigmatism_axis" type="number" min="0" max="180" step="1" required />
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label for="pachy_x">Pachymetry Thinnest X Coordinate</label>
              <input id="pachy_x" name="pachy_x" type="number" step="0.01" required />
            </div>
            <div class="field">
              <label for="pachy_y">Pachymetry Thinnest Y Coordinate</label>
              <input id="pachy_y" name="pachy_y" type="number" step="0.01" required />
            </div>
          </div>
          <div class="field">
            <label>Topography Grid Upload</label>
            <div class="upload-grid">
              <div class="upload-card">
                <label for="anterior_map">Anterior Map</label>
                <input id="anterior_map" name="anterior_map" type="file" accept="image/*" required onchange="showPreview(this,'anterior_preview')" />
                <small>High-resolution anterior map</small>
                <div id="anterior_preview" class="preview"><div class="placeholder">No image selected</div></div>
              </div>
              <div class="upload-card">
                <label for="axial_map">Axial Map</label>
                <input id="axial_map" name="axial_map" type="file" accept="image/*" required onchange="showPreview(this,'axial_preview')" />
                <small>Axial curvature map</small>
                <div id="axial_preview" class="preview"><div class="placeholder">No image selected</div></div>
              </div>
              <div class="upload-card">
                <label for="pachymetry_map">Pachymetry Map</label>
                <input id="pachymetry_map" name="pachymetry_map" type="file" accept="image/*" required onchange="showPreview(this,'pachymetry_preview')" />
                <small>Corneal thickness map</small>
                <div id="pachymetry_preview" class="preview"><div class="placeholder">No image selected</div></div>
              </div>
              <div class="upload-card">
                <label for="posterior_map">Posterior Map</label>
                <input id="posterior_map" name="posterior_map" type="file" accept="image/*" required onchange="showPreview(this,'posterior_preview')" />
                <small>Posterior elevation map</small>
                <div id="posterior_preview" class="preview"><div class="placeholder">No image selected</div></div>
              </div>
            </div>
          </div>
          <button class="btn" type="submit">Run Clinical Assessment</button>
        </form>
      </div>
      <div class="card">
        <h3 style="margin-top:0">Clinical Guidelines</h3>
        <ul>
          <li>Use this interface as a decision-support overlay rather than a sole diagnostic authority.</li>
          <li>Ensure all four corneal maps are supplied as distinct, high-quality images.</li>
          <li>Review the Grad-CAM overlay for structural regions that most strongly influenced the outcome.</li>
        </ul>
      </div>
    </div>
  </div>
  <script>
    function showPreview(input, targetId) {
      const preview = document.getElementById(targetId);
      if (!preview) return;
      if (!input.files || !input.files[0]) {
        preview.innerHTML = '<div class="placeholder">No image selected</div>';
        return;
      }
      const file = input.files[0];
      const reader = new FileReader();
      reader.onload = function (event) {
        preview.innerHTML = '<img src="' + event.target.result + '" alt="Uploaded preview" />';
      };
      reader.readAsDataURL(file);
    }
  </script>
</body>
</html>
"""


RESULT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EyeNova • Diagnostic Result</title>
  <style>
    :root { --bg:#f4f7fb; --panel:#ffffff; --ink:#10324a; --muted:#617688; --accent:#1f5d8a; --accent2:#0b6ea8; --line:#dbe4ee; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Segoe UI,Roboto,Arial,sans-serif; background:linear-gradient(135deg,#eef4fb,#f7fbff); color:var(--ink); }
    .shell { max-width:1180px; margin:0 auto; padding:28px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:24px; padding:24px; box-shadow:0 12px 30px rgba(16,50,74,0.06); }
    .header { display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:18px; }
    .status { display:inline-block; padding:8px 12px; border-radius:999px; font-weight:700; background:#e7f4ff; color:var(--accent); }
    .metrics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:16px 0 22px; }
    .metric { padding:14px; border-radius:14px; background:#f8fbff; border:1px solid var(--line); }
    .layout { display:grid; grid-template-columns:1.1fr .9fr; gap:22px; align-items:start; }
    .heatmap { border-radius:18px; border:1px solid var(--line); padding:14px; background:#fbfdff; }
    .heatmap img { width:100%; display:block; border-radius:12px; }
    .btn { width:auto; display:inline-block; text-decoration:none; background:linear-gradient(135deg,var(--accent),var(--accent2)); color:white; padding:12px 18px; border-radius:12px; font-weight:700; }
    @media (max-width:900px){ .layout,.metrics{grid-template-columns:1fr;} }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <div class="header">
        <div>
          <h1 style="margin:0 0 6px">Diagnostic Result</h1>
          <p style="margin:0;color:var(--muted)">EyeNova multimodal assessment generated from the uploaded corneal maps and clinical inputs.</p>
        </div>
        <a class="btn" href="/">Reset / New Scan</a>
      </div>
      <div class="status">Status: {{ status }}</div>
      <div class="metrics">
        <div class="metric"><strong>Confidence</strong><div>{{ confidence }}%</div></div>
        <div class="metric"><strong>Model</strong><div>ViT + Clinical Fusion</div></div>
      </div>
      <div class="layout">
        <div class="heatmap">
          <h3 style="margin-top:0">Grad-CAM Interpretability</h3>
          <img src="data:image/png;base64,{{ heatmap_b64 }}" alt="Grad-CAM heatmap" />
        </div>
        <div class="card" style="padding:18px">
          <h3 style="margin-top:0">Clinical Interpretation</h3>
          <p>The overlay highlights regions of the stitched topography view that were most influential in the model's prediction. A stronger signal in the central and paracentral corneal regions may be consistent with keratoconic morphology.</p>
          <p style="color:var(--muted)">This output is intended to support clinician review and should be interpreted alongside standard clinical examination and tomography findings.</p>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.get('/')
def index():
    return render_template_string(HTML_FORM)


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
        return render_template_string(RESULT_TEMPLATE, status=status, confidence=f'{confidence:.1f}', heatmap_b64=heatmap_b64)
    except Exception as exc:
        return render_template_string("<h2>Assessment Error</h2><p>{{ error }}</p><a href='/'>Return</a>", error=str(exc))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
