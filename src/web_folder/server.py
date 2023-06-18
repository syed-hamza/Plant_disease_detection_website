from flask import Flask, flash, request, redirect, url_for, render_template, session
import urllib.request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import sys
import torch
from torch import nn
sys.path.append('../utils')
from model import CNN
import torchvision.transforms as transforms

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
file_path = None
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
size = 128
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Healthy","Powdery","Rust"]
model_path = "../../models/model.pth"
model = torch.load(model_path).to(device)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        flash("Image uploaded and displayed below")
        session['filename'] = filename
        return render_template('home.html', filename=filename)
    else:
        flash('Allowed image types are png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/predict')
def predict():
    filename = session.get('filename')
    if filename:
        image = Image.open('static/uploads/' + filename)
        image = torch.unsqueeze(transform(image),0).to(device)
        preds = model(image)
        set = classes[torch.argmax(preds)]
        # Perform prediction logic
        return set # Replace this with your actual prediction result
    else:
        return "Choose file"
@app.route('/reload', methods=['POST'])
def reload_page():
    filename = session.get('filename')
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        session.pop('filename', None)
        flash("Uploaded image removed")
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)
