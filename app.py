from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import requests
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_captions(image_path, num_captions=3):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    pixel_values = image.float() / 255.0

    generated_captions = []

    for _ in range(num_captions):
        generated_ids = model_raw.generate(pixel_values, max_new_tokens=30, do_sample=True)
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_captions.append(generated_caption)

    image_np = pixel_values.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig('static/result.png')  # Save the result image
    plt.close()

    return generated_captions

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    num_captions = int(request.form['num_captions'])

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        captions = generate_captions(file_path, num_captions)

        return render_template('result.html', image_path=f'static/result.png', captions=captions)

    return render_template('index.html', error='Invalid file format or upload failed.')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)
