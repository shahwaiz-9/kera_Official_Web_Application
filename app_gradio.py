import os
import gradio as gr
import numpy as np
from PIL import Image
import torch
from app import EyeNovaApp

# Initialize the engine
engine = EyeNovaApp()
try:
    engine.load_artifacts()
except Exception as e:
    print(f"Error loading artifacts: {e}")

def predict_gradio(age, gender, astigmatism_value, astigmatism_axis, pachy_x, pachy_y, ant_img, ax_img, pachy_img, post_img):
    if ant_img is None or ax_img is None or pachy_img is None or post_img is None:
        return "Error: All 4 topography maps are required.", "", None, None
    
    # Prepare payload
    payload = {
        'age': age,
        'gender': gender,
        'astigmatism_value': astigmatism_value,
        'astigmatism_axis': astigmatism_axis,
        'pachy_x': pachy_x,
        'pachy_y': pachy_y,
    }
    
    # Process images (converting inputs, which could be NumPy arrays from Gradio, to PIL Images)
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
    
    # Run inference
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

# Gradio layout configuration
with gr.Blocks(theme=gr.themes.Soft()) as demo:
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
    demo.launch(server_name="0.0.0.0", server_port=7860)
