import base64
import os
import tempfile
from flask import Flask, render_template, render_template_string, request
from gradio_client import Client, handle_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Connect to the Hugging Face Space API
HF_SPACE_ID = "Shahwaiz-9/keratoconus-multimodal-app"

@app.get('/')
def index():
    return render_template('index.html')

@app.post('/predict')
def predict():
    temp_files = []
    try:
        # 1. Parse tabular inputs
        required_fields = ['age', 'gender', 'astigmatism_value', 'astigmatism_axis', 'pachy_x', 'pachy_y']
        payload = {}
        for field in required_fields:
            value = request.form.get(field, '').strip()
            if not value:
                raise ValueError(f'Missing required field: {field}')
            payload[field] = value

        # 2. Get uploaded scan files
        scan_fields = ['anterior_map', 'axial_map', 'pachymetry_map', 'posterior_map']
        scan_paths = []
        for field in scan_fields:
            file_storage = request.files.get(field)
            if not file_storage or file_storage.filename == '':
                raise ValueError(f'All four topography maps must be uploaded. Missing: {field}')
            
            # Save file storage object to a temporary file
            suffix = os.path.splitext(file_storage.filename)[1] or '.png'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            file_storage.save(temp_file.name)
            temp_file.close()
            temp_files.append(temp_file.name)
            scan_paths.append(temp_file.name)

        # 3. Call the Hugging Face Space API
        client = Client(HF_SPACE_ID)
        
        # Call the Gradio prediction function
        result = client.predict(
            float(payload['age']),
            str(payload['gender']),
            float(payload['astigmatism_value']),
            float(payload['astigmatism_axis']),
            float(payload['pachy_x']),
            float(payload['pachy_y']),
            handle_file(scan_paths[0]), # Anterior
            handle_file(scan_paths[1]), # Axial
            handle_file(scan_paths[2]), # Pachymetry
            handle_file(scan_paths[3]), # Posterior
            fn_index=0
        )

        # Gradio return value format: (status, confidence, stitched_img_path, heatmap_img_path)
        status, confidence_str, stitched_img_path, heatmap_img_path = result

        # 4. Convert output images to Base64 for the original results template
        with open(stitched_img_path, "rb") as image_file:
            original_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        with open(heatmap_img_path, "rb") as image_file:
            heatmap_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Clear confidence percentage symbol if present in string for formatting
        confidence = confidence_str.replace('%', '').strip()

        return render_template(
            'result.html',
            status=status,
            confidence=confidence,
            original_b64=original_b64,
            heatmap_b64=heatmap_b64
        )

    except Exception as exc:
        return render_template_string("<h2>Assessment Error</h2><p>{{ error }}</p><a href='/'>Return</a>", error=str(exc))
    
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except OSError:
                pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
