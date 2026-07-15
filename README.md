---
title: Keratoconus Multimodal Web Application
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Keratoconus Multimodal Web Application

This is a late-fusion multimodal deep learning application that assists in the diagnosis of Keratoconus using:
1. **ViT (Vision Transformer) Stream**: For corneal topography maps.
2. **Tabular MLP Stream**: For clinical parameters (age, gender, astigmatism, pachymetry).

The app is built using Flask, PyTorch, Hugging Face Transformers, and scikit-learn.

## Local Running

To run this application locally:

1. Build the Docker image:
   ```bash
   docker build -t keratoconus-app .
   ```

2. Run the container:
   ```bash
   docker run -p 7860:7860 keratoconus-app
   ```

3. Open your browser and navigate to `http://localhost:7860`.
