
import os
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io

from utils import preprocess_image, load_model, predict_image

app = Flask(__name__, template_folder='templates')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

DEFAULT_CLASS_NAMES = [
    'Airliner', 'Balloon', 'Helicopter', 'Dirigible', 
    'Rockets', 'Spaceshuttle', 'arplane'
]

# Global variables for loaded models
MODELS = {}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
MODEL_CONFIGS = [
    {
        'name': 'CNN Model 1',
        'path': 'models/model_cnn1.pth',
        'type': 'cnn',
        'num_classes': 7,
        'image_size': 64
    },
    {
        'name': 'MLP Model 1',
        'path': 'models/model_mlp1.pth',
        'type': 'mlp',
        'num_classes': 7,
        'image_size': 64
    },
    {
        'name': 'AlexNet',
        'path': 'models/model_alexnet1.pth',
        'type': 'alexnet',
        'num_classes': 7,
        'image_size': 224
    },
    {
        'name': 'ResNet50',
        'path': 'models/model_resnet1.pth',
        'type': 'resnet',
        'num_classes': 7,
        'image_size': 224
    },
]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_all_models():
    global MODELS
    for config in MODEL_CONFIGS:
        model_path = os.path.join(os.path.dirname(__file__), config['path'])
        if os.path.exists(model_path):
            model = load_model(
            model_path,
            config['type'],
            config['num_classes'],
            DEVICE
             )
            print(f"Loaded model: {config['name']}")
            MODELS[config['name']] = {
                'model': model,
                'type': config['type'],
                'image_size': config.get('image_size', 64), 
                'config': config
            }
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    if not MODELS:
        print("Warning: No models loaded. Please check model paths.")


@app.route('/')
def index():
    return render_template('index.html')


@app.post('/predict')
def predict():
    file = request.files['image']
    model_name = request.form.get('model')

    # Read image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
    # Get model and configuration
    model_data = MODELS[model_name]
    model = model_data['model']
    model_type = model_data['type']
    image_size = model_data['image_size']
        
    # Preprocess image
    image_tensor = preprocess_image(image, image_size=image_size)
        
    # Make prediction
    result = predict_image(model, image_tensor, DEFAULT_CLASS_NAMES, DEVICE, model_type)
        
    # response
    response = {
        'success': True,
        'model_used': model_name,
        'prediction': result['predicted_class_name'],
        'confidence': round(result['confidence'], 4),
        'confidence_percent': round(result['confidence'] * 100, 2),
        'all_predictions': [
            {
                'class': pred['class'],
                'probability': round(pred['probability'], 4),
                'probability_percent': round(pred['probability'] * 100, 2)
            }
            for pred in sorted(result['all_predictions'], 
                             key=lambda x: x['probability'], reverse=True)
        ]
    }
        
    return jsonify(response)




if __name__ == '__main__':
    print("Loading models...")
    load_all_models()
    print(f"Loaded {len(MODELS)} model(s)")
    print("Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5001)

