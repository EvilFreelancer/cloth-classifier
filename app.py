import gc
import os
import json
import logging
import torch
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from torchvision import transforms
from PIL import Image

from cloth_classifier.models import AdvancedNet, SimpleNet, RhombusNet
from cloth_classifier.preprocess_image import get_transformation

logger = logging.getLogger('Api')

# Port and binding
api_port = int(os.getenv('APP_PORT', 5000))
logger.info(f'API port is: {api_port}')
api_bind = os.getenv('APP_BIND', '0.0.0.0')
logger.info(f'API bind to: {api_bind}')

# Init application and dependencies
app = Flask(__name__)
model = SimpleNet(input_size=4096).to('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('./models/cloth_model_simple.pth'))
model.eval()

CLASSES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]

transform = get_transformation(64, 64)


def inference(img):
    img = img.view(img.shape[0], -1)  # Convert 2D image to 1D vector
    prediction = model(img)
    proba = torch.exp(prediction)
    return proba


def format_prediction(prediction):
    proba = torch.exp(prediction)
    class_label = CLASSES[proba.argmax()]
    return {"class": int(proba.argmax()), "class_label": class_label, "probability": proba.max().item()}


@app.errorhandler(HTTPException)
def handle_exception(e):
    logger.exception(e)
    gc.collect()
    response = e.get_response()
    response.data = json.dumps({"code": e.code, "name": e.name, "description": e.description})
    response.content_type = "application/json"
    return response


@app.route('/')
@app.route('/index')
def index():
    return "Cloth Classifier API"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        try:
            img = Image.open(file).convert('L')  # Convert image to grayscale
            img = transform(img)
            img = img.unsqueeze(0)  # Add batch dimension
            img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
            prediction = inference(img)
            result = format_prediction(prediction)
            return jsonify(result), 200
        except Exception as e:
            logger.exception(e)
            return jsonify({'message': 'Error processing image'}), 500


if __name__ == "__main__":
    app.run(host=api_bind, port=api_port, debug=True, use_reloader=True)
