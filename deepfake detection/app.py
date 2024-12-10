from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Folder to store uploaded files temporarily
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions (you can add video formats as well)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Setup MTCNN for face detection and load your pre-trained face classification model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=DEVICE)  # MTCNN for face detection
model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE).to(DEVICE)

# Load model checkpoint
checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_face(input_image):
    """ Detects the face in the image and preprocesses it for the model. """
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    
    face = face.unsqueeze(0)  # Add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    face = face.to(DEVICE).float() / 255.0  # Normalize the face
    return face

def classify_face(face_tensor):
    """ Run the face tensor through the model and return predictions. """
    with torch.no_grad():
        output = torch.sigmoid(model(face_tensor).squeeze(0))
        real_prob = 1 - output.item()
        fake_prob = output.item()
        prediction = "real" if real_prob > fake_prob else "fake"
    return prediction, real_prob, fake_prob

def generate_explainability_map(face_tensor, prediction):
    """ Generates Grad-CAM visualization for the given prediction. """
    target_layers = [model.logits]  # Final classification layer
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    targets = [ClassifierOutputTarget(0)] if prediction == "real" else [ClassifierOutputTarget(1)]
    
    grayscale_cam = cam(input_tensor=face_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Get the Grad-CAM map for the first image in the batch
    
    face_image = face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor to image
    face_image = (face_image * 255).astype(np.uint8)  # Convert back to 0-255 scale
    
    visualization = show_cam_on_image(face_image, grayscale_cam, use_rgb=True)
    return visualization

@app.route('/classify', methods=['POST'])
def classify_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load image and preprocess
            img = Image.open(filepath).convert('RGB')
            face_tensor = preprocess_face(img)

            # Classify the face
            prediction, real_prob, fake_prob = classify_face(face_tensor)

            # Generate explainability map
            explainability_map = generate_explainability_map(face_tensor, prediction)

            # Convert explainability map to a format that can be returned as a URL
            map_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'explainability_map.png')
            Image.fromarray(explainability_map).save(map_filepath)

            return jsonify({
                "prediction": prediction,
                "real_confidence": real_prob,
                "fake_confidence": fake_prob,
                "explainability_map_url": map_filepath
            })
        else:
            return jsonify({"error": "File type not supported"}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to classify file."}), 500

if __name__ == "__main__":
    app.run(debug=True)
